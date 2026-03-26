#!/usr/bin/env python3
"""SAC training for MuJoCo locomotion via mujoco-torch + TorchRL.

Uses TorchRL replay buffer, loss module, and transforms.
The zoo envs use vmap internally for fast batched simulation.

Usage::

    python examples/train_sac.py --env halfcheetah --num_envs 256
    python examples/train_sac.py --env humanoid --num_envs 256
    python examples/train_sac.py --env ant --num_envs 256
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    RewardSum,
    StepCounter,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
)
from torchrl.objectives import SACLoss
from torchrl.objectives.utils import SoftUpdate
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

from mujoco_torch.zoo import ENVS

# ------------------------------------------------------------------
# Numeric health checks — NaN, Inf, and extreme values
# ------------------------------------------------------------------

# Absolute threshold: any value beyond this is a physics blow-up, not a
# legitimate reward or observation.
_EXTREME_THRESHOLD = 1e10


def _check_td_health(td, label, keys=None):
    """Check a TensorDict for NaN, Inf, or extreme values.

    Returns list of (key, nan_count, inf_count, extreme_count, shape, max_abs, min_val, max_val).
    """
    issues = []
    items = td.items(include_nested=True) if keys is None else [(k, td[k]) for k in keys if k in td.keys(True)]
    for key, val in items:
        if not isinstance(val, torch.Tensor) or not val.is_floating_point():
            continue
        nan_count = val.isnan().sum().item()
        inf_count = val.isinf().sum().item()
        finite = val[val.isfinite()]
        extreme_count = (finite.abs() > _EXTREME_THRESHOLD).sum().item() if finite.numel() > 0 else 0
        if nan_count > 0 or inf_count > 0 or extreme_count > 0:
            max_abs = finite.abs().max().item() if finite.numel() > 0 else float("nan")
            min_val = finite.min().item() if finite.numel() > 0 else float("nan")
            max_val = finite.max().item() if finite.numel() > 0 else float("nan")
            issues.append((key, nan_count, inf_count, extreme_count, val.shape, max_abs, min_val, max_val))
    return issues


def _check_params_nan(module, label):
    """Check module parameters and gradients for NaN/Inf."""
    issues = []
    for name, p in module.named_parameters():
        if p.isnan().any():
            issues.append((name, "param", p.isnan().sum().item(), p.shape))
        if p.isinf().any():
            issues.append((name, "param_inf", p.isinf().sum().item(), p.shape))
        if p.grad is not None:
            if p.grad.isnan().any():
                issues.append((name, "grad", p.grad.isnan().sum().item(), p.grad.shape))
            if p.grad.isinf().any():
                issues.append((name, "grad_inf", p.grad.isinf().sum().item(), p.grad.shape))
    return issues


def _fatal(stage, collect_iter, collected_frames, detail, save_path=None, save_data=None):
    """Print detailed diagnosis and crash."""
    msg = [
        "",
        "=" * 70,
        f"FATAL: Bad numerics at stage '{stage}'",
        f"  collect_iter={collect_iter}  collected_frames={collected_frames}",
        "=" * 70,
    ]
    if isinstance(detail, list):
        for item in detail:
            msg.append(f"  {item}")
    else:
        msg.append(f"  {detail}")
    msg.append("=" * 70)
    full_msg = "\n".join(msg)
    print(full_msg, flush=True)

    if save_path is not None and save_data is not None:
        try:
            torch.save(save_data, save_path)
            print(f"  Diagnostic data saved to {save_path}", flush=True)
        except Exception as e:
            print(f"  Failed to save diagnostic: {e}", flush=True)

    raise RuntimeError(full_msg)


def _format_health_issues(issues):
    """Format health check issues into human-readable lines."""
    lines = []
    for key, nan_c, inf_c, ext_c, shape, _max_abs, min_val, max_val in issues:
        parts = [f"{key} shape={shape}:"]
        if nan_c:
            parts.append(f"NaN={nan_c}")
        if inf_c:
            parts.append(f"Inf={inf_c}")
        if ext_c:
            parts.append(f"extreme(>{_EXTREME_THRESHOLD:.0e})={ext_c}")
        parts.append(f"range=[{min_val:.2e}, {max_val:.2e}]")
        lines.append(" ".join(parts))
    return lines


# ------------------------------------------------------------------
# Environment factories
# ------------------------------------------------------------------


def make_env(env_name, num_envs, device, frame_skip, compile_step=False, compile_kwargs=None, obs_norm_num_iter=50):
    cls = ENVS[env_name]
    base = cls(num_envs=num_envs, device=device, frame_skip=frame_skip, compile_step=compile_step, compile_kwargs=compile_kwargs)
    env = TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            ObservationNorm(in_keys=["observation"], standard_normal=True),
            StepCounter(max_steps=1000),
            RewardSum(),
        ),
    )
    print(f"  ObservationNorm: computing stats ({obs_norm_num_iter} rollouts, {num_envs} envs)...", flush=True)
    t_init = time.perf_counter()
    env.transform[1].init_stats(obs_norm_num_iter, reduce_dim=(0, 1), cat_dim=0)
    obs_norm = env.transform[1]
    elapsed = time.perf_counter() - t_init
    print(
        f"  ObservationNorm: done in {elapsed:.1f}s"
        f"  loc=[{obs_norm.loc.min():.2f}, {obs_norm.loc.max():.2f}]"
        f"  scale=[{obs_norm.scale.min():.2f}, {obs_norm.scale.max():.2f}]",
        flush=True,
    )
    return env


def make_eval_env(env_name, device, frame_skip, logger, obs_norm_td=None):
    cls = ENVS[env_name]
    base = cls(num_envs=1, device=device, frame_skip=frame_skip)
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    env = TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            obs_norm,
            StepCounter(max_steps=1000),
            RewardSum(),
            PixelRenderTransform(out_keys=["pixels"]),
            VideoRecorder(
                logger=logger,
                tag="eval_video",
                skip=1,
                make_grid=False,
            ),
        ),
    )
    if obs_norm_td is not None:
        obs_norm.loc = obs_norm_td["loc"].clone()
        obs_norm.scale = obs_norm_td["scale"].clone()
        print(f"  Eval ObservationNorm: initialized={obs_norm.initialized} "
              f"loc.shape={obs_norm.loc.shape} scale.shape={obs_norm.scale.shape}", flush=True)
    return env


# ------------------------------------------------------------------
# Model factories
# ------------------------------------------------------------------


def make_actor(obs_dim, act_dim, device):
    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
            device=device,
        ),
        NormalParamExtractor(),
    )
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    return ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


class QValueMLP(nn.Module):
    """Q-network that concatenates observation and action."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, device=None):
        super().__init__()
        self.mlp = MLP(
            in_features=obs_dim + act_dim,
            out_features=1,
            num_cells=[hidden_dim, hidden_dim],
            activation_class=nn.ReLU,
            device=device,
        )

    def forward(self, obs, action):
        return self.mlp(torch.cat([obs, action], dim=-1))


def make_qvalue(obs_dim, act_dim, device):
    return TensorDictModule(
        QValueMLP(obs_dim, act_dim, device=device),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )


# ------------------------------------------------------------------
# Eval
# ------------------------------------------------------------------


def run_eval(eval_env, actor, iteration, logger, max_steps=1000):
    actor.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = eval_env.rollout(
            max_steps=max_steps,
            policy=actor,
            auto_reset=True,
            break_when_any_done=True,
        )
    actor.train()

    ep_reward = rollout["next", "episode_reward"][:, -1].mean().item()

    log_dict = {
        "eval/episode_reward": ep_reward,
        "eval/mean_step_reward": rollout["next", "reward"].mean().item(),
        "iteration": iteration,
    }

    for t in eval_env.transform:
        if isinstance(t, VideoRecorder) and t.obs:
            try:
                import wandb

                vid = torch.stack(t.obs, 0).unsqueeze(0).cpu()
                log_dict["eval_video"] = wandb.Video(
                    vid,
                    fps=30,
                    format="mp4",
                )
            except Exception:
                pass
            t.obs = []
            t.count = 0

    logger.experiment.log(log_dict)
    return ep_reward


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def train(args):
    env_device = args.device
    train_device = args.train_device or env_device

    # --- Envs ---
    compile_kwargs = {}
    if args.compile_mode:
        compile_kwargs["mode"] = args.compile_mode
    train_env = make_env(
        args.env,
        args.num_envs,
        env_device,
        args.frame_skip,
        compile_step=args.compile,
        compile_kwargs=compile_kwargs or None,
    )
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    act_dim = train_env.action_spec.shape[-1]

    # --- Logger ---
    logger = WandbLogger(
        exp_name=f"{args.env}_sac",
        project=args.wandb_project,
        config=vars(args),
    )
    # Share obs normalization stats from train env to eval env
    obs_norm_td = train_env.transform[1].state_dict()
    eval_env = make_eval_env(args.env, train_device, args.frame_skip, logger, obs_norm_td=obs_norm_td)

    # --- Models (on train device) ---
    actor = make_actor(obs_dim, act_dim, train_device)
    qvalue = make_qvalue(obs_dim, act_dim, train_device)

    # --- SAC loss ---
    # target_entropy = -act_dim (standard heuristic)
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        action_spec=train_env.action_spec.to(train_device),
    )
    loss_module.make_value_estimator(
        loss_module.default_value_estimator,
        gamma=args.gamma,
    )

    # Target network soft-update
    target_updater = SoftUpdate(loss_module, tau=args.tau)

    # Separate optimizers (standard for SAC)
    actor_optim = torch.optim.Adam(
        list(loss_module.actor_network_params.values(True, True)),
        lr=args.lr,
    )
    critic_optim = torch.optim.Adam(
        list(loss_module.qvalue_network_params.values(True, True)),
        lr=args.lr,
    )
    alpha_optim = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=args.lr,
    )

    # --- Replay buffer (on train device) ---
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=train_device),
        batch_size=args.batch_size,
    )

    # --- Collector ---
    # Env runs on env_device; collected data is stored on train_device
    collector = SyncDataCollector(
        train_env,
        actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=env_device,
        storing_device=train_device,
    )

    print(
        f"SAC [{args.env}] obs={obs_dim} act={act_dim}\n"
        f"  env_device={env_device} train_device={train_device}\n"
        f"  num_envs={args.num_envs} frames_per_batch={args.frames_per_batch} "
        f"buffer_size={args.buffer_size} batch_size={args.batch_size}\n"
        f"  gamma={args.gamma} tau={args.tau} lr={args.lr}",
        flush=True,
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    collected_frames = 0
    num_updates = 0

    diag_dir = Path("nan_diagnostics")
    diag_dir.mkdir(exist_ok=True)

    check_keys = [
        ("observation",),
        ("action",),
        ("next", "observation"),
        ("next", "reward"),
        ("next", "done"),
    ]

    for collect_iter, batch in enumerate(collector):
        collected_frames += batch.numel()

        # === Stage 1: Check collected batch for NaN / Inf / extreme values ===
        batch_issues = _check_td_health(batch, "batch", keys=check_keys)
        if batch_issues:
            _fatal(
                "collected_batch",
                collect_iter,
                collected_frames,
                _format_health_issues(batch_issues),
                save_path=diag_dir / f"batch_iter{collect_iter}.pt",
                save_data={
                    "batch": batch.detach().cpu(),
                    "collect_iter": collect_iter,
                },
            )

        # Store in replay buffer
        buffer.extend(batch.reshape(-1))

        # Skip updates until we have enough data
        if collected_frames < args.learning_starts:
            if (collect_iter + 1) % 100 == 0:
                print(
                    f"  collecting... {collected_frames}/{args.learning_starts}",
                    flush=True,
                )
            continue

        # --- Update ---
        for update_idx in range(args.utd_ratio):
            sample = buffer.sample()

            # === Stage 2: Check replay sample ===
            sample_issues = _check_td_health(sample, "sample")
            if sample_issues:
                _fatal(
                    "replay_sample",
                    collect_iter,
                    collected_frames,
                    _format_health_issues(sample_issues),
                    save_path=diag_dir / f"sample_iter{collect_iter}_u{update_idx}.pt",
                    save_data={"sample": sample.detach().cpu()},
                )

            loss_vals = loss_module(sample)

            # === Stage 3: Check losses for NaN / Inf / extreme ===
            loss_issues = []
            for lk in ["loss_qvalue", "loss_actor", "loss_alpha"]:
                v = loss_vals[lk]
                if v.isnan().any() or v.isinf().any() or v.abs() > _EXTREME_THRESHOLD:
                    loss_issues.append(f"{lk}={v.item():.6e}")
            if loss_issues:
                with torch.no_grad():
                    actor_out = actor(sample)
                    q_out = qvalue(sample)
                obs = sample["observation"]
                act = sample["action"]
                rew = sample["next", "reward"]
                _fatal(
                    "loss_computation",
                    collect_iter,
                    collected_frames,
                    loss_issues
                    + [
                        (
                            f"actor loc: [{actor_out['loc'].min():.2e}, {actor_out['loc'].max():.2e}]"
                            f"  nan={actor_out['loc'].isnan().sum().item()}"
                        ),
                        (
                            f"actor scale: [{actor_out['scale'].min():.2e},"
                            f" {actor_out['scale'].max():.2e}]"
                            f"  nan={actor_out['scale'].isnan().sum().item()}"
                        ),
                        (
                            f"q_value: [{q_out['state_action_value'].min():.2e},"
                            f" {q_out['state_action_value'].max():.2e}]"
                            f"  nan={q_out['state_action_value'].isnan().sum().item()}"
                        ),
                        f"alpha={loss_module.log_alpha.exp().item():.6e}  log_alpha={loss_module.log_alpha.item():.6e}",
                        (
                            f"obs: [{obs.min():.2e}, {obs.max():.2e}]"
                            f"  nan={obs.isnan().sum().item()}"
                            f"  |obs|>1e6: {(obs.abs() > 1e6).sum().item()}"
                        ),
                        f"action: [{act.min():.2e}, {act.max():.2e}]  nan={act.isnan().sum().item()}",
                        (
                            f"reward: [{rew.min():.2e}, {rew.max():.2e}]"
                            f"  nan={rew.isnan().sum().item()}"
                            f"  |r|>1e6: {(rew.abs() > 1e6).sum().item()}"
                        ),
                    ],
                    save_path=diag_dir / f"loss_iter{collect_iter}_u{update_idx}.pt",
                    save_data={
                        "sample": sample.detach().cpu(),
                        "loss_vals": {k: v.detach().cpu() for k, v in loss_vals.items()},
                        "log_alpha": loss_module.log_alpha.detach().cpu(),
                        "actor_out": {k: v.detach().cpu() for k, v in actor_out.items() if isinstance(v, torch.Tensor)},
                    },
                )

            # Zero all grads, backward all losses, then step
            critic_optim.zero_grad()
            actor_optim.zero_grad()
            alpha_optim.zero_grad()

            loss_vals["loss_qvalue"].backward(retain_graph=True)
            loss_vals["loss_actor"].backward(retain_graph=True)
            loss_vals["loss_alpha"].backward()

            nn.utils.clip_grad_norm_(
                list(loss_module.qvalue_network_params.values(True, True)),
                1.0,
            )
            nn.utils.clip_grad_norm_(
                list(loss_module.actor_network_params.values(True, True)),
                1.0,
            )

            # === Stage 4: Check gradients ===
            param_issues = _check_params_nan(loss_module, "loss_module")
            if param_issues:
                _fatal(
                    "gradients",
                    collect_iter,
                    collected_frames,
                    [f"{name} ({kind}): count={cnt} shape={s}" for name, kind, cnt, s in param_issues],
                    save_path=diag_dir / f"grad_iter{collect_iter}_u{update_idx}.pt",
                    save_data={
                        "params": {n: p.detach().cpu() for n, p in loss_module.named_parameters()},
                        "grads": {
                            n: p.grad.detach().cpu() for n, p in loss_module.named_parameters() if p.grad is not None
                        },
                    },
                )

            critic_optim.step()
            actor_optim.step()
            alpha_optim.step()

            # === Stage 5: Check params after optimizer step ===
            post_issues = _check_params_nan(loss_module, "post_step")
            if post_issues:
                _fatal(
                    "post_optimizer_step",
                    collect_iter,
                    collected_frames,
                    [f"{name} ({kind}): count={cnt} shape={s}" for name, kind, cnt, s in post_issues],
                    save_path=diag_dir / f"param_iter{collect_iter}_u{update_idx}.pt",
                    save_data={
                        "params": {n: p.detach().cpu() for n, p in loss_module.named_parameters()},
                    },
                )

            # Target update
            target_updater.step()
            num_updates += 1

        # --- Logging ---
        if (collect_iter + 1) % args.log_interval == 0:
            ep_done = batch["next", "done"].squeeze(-1)
            ep_rews = batch["next", "episode_reward"][ep_done]
            mean_ep = ep_rews.mean().item() if ep_rews.numel() > 0 else float("nan")
            if mean_ep == mean_ep:
                best_reward = max(best_reward, mean_ep)

            alpha = loss_module.log_alpha.exp().item()

            logger.experiment.log(
                {
                    "train/mean_ep_reward": mean_ep,
                    "train/best_ep_reward": best_reward,
                    "train/loss_actor": loss_vals["loss_actor"].item(),
                    "train/loss_qvalue": loss_vals["loss_qvalue"].item(),
                    "train/loss_alpha": loss_vals["loss_alpha"].item(),
                    "train/alpha": alpha,
                    "train/num_updates": num_updates,
                    "collected_frames": collected_frames,
                    "iteration": collect_iter,
                }
            )

            elapsed = time.perf_counter() - t0
            fps = collected_frames / elapsed
            print(
                f"  iter {collect_iter + 1} | frames={collected_frames} | "
                f"ep_reward={mean_ep:.1f} | best={best_reward:.1f} | "
                f"alpha={alpha:.3f} | fps={fps:.0f}",
                flush=True,
            )

        if (collect_iter + 1) % args.eval_interval == 0:
            eval_r = run_eval(eval_env, actor, collect_iter, logger)
            print(f"    [EVAL] ep_reward={eval_r:.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"Done. {collected_frames} frames in {elapsed:.0f}s. Best ep reward: {best_reward:.1f}",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="SAC for MuJoCo locomotion")

    # Env
    p.add_argument(
        "--env",
        type=str,
        default="halfcheetah",
        choices=list(ENVS.keys()),
    )
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--frame_skip", type=int, default=5)

    # Collection
    p.add_argument(
        "--frames_per_batch", type=int, default=None, help="Frames per collection batch (default: num_envs * 1000)"
    )
    p.add_argument("--total_frames", type=int, default=3_000_000)
    p.add_argument("--learning_starts", type=int, default=25000)

    # SAC
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--buffer_size", type=int, default=1_000_000)
    p.add_argument("--utd_ratio", type=int, default=1)

    # Logging
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--wandb_project", type=str, default="mujoco-torch-zoo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Env/collection device (default: cuda)")
    p.add_argument("--train_device", type=str, default=None, help="Training device (default: same as --device)")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--compile_mode", type=str, default=None,
                   help="torch.compile mode (e.g. 'reduce-overhead', 'max-autotune')")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.frames_per_batch is None:
        args.frames_per_batch = args.num_envs * 1000

    train(args)


if __name__ == "__main__":
    main()
