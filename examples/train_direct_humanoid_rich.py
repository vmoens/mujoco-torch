#!/usr/bin/env python3
"""Direct backprop optimization of Humanoid with rich observations.

Same as train_direct_humanoid.py but uses HumanoidRichEnv which includes
cinert, cvel, and qfrc_actuator in the observation space (336-dim instead
of 53-dim), matching Gymnasium Humanoid-v5's observation structure.

Usage::

    python examples/train_direct_humanoid_rich.py
    python examples/train_direct_humanoid_rich.py --device cuda --num_envs 128
"""

import argparse
import time

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

import mujoco_torch
from mujoco_torch._src.log import logger as mjt_logger
from mujoco_torch.zoo.humanoid_rich import HumanoidRichEnv

# ------------------------------------------------------------------
# Differentiable training env
# ------------------------------------------------------------------


class SmoothHumanoidRichEnv(HumanoidRichEnv):
    """HumanoidRich with smooth reward and no hard termination."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mx = self.mx
        frame_skip = self.FRAME_SKIP
        _step_fn = lambda d: mujoco_torch.step(mx, d, fixed_iterations=True)  # noqa: E731
        if self.num_envs == 1:

            def _multi_step(d):
                for _ in range(frame_skip):
                    d = _step_fn(d)
                return d

            self._physics_step = _multi_step
        else:
            _vmap_step = torch.vmap(_step_fn)

            def _vmap_multi_step(d):
                for _ in range(frame_skip):
                    d = _vmap_step(d)
                return d

            self._physics_step = _vmap_multi_step

    def _compute_reward(self, qpos_before, action):
        forward_vel = (self._dx.qpos[..., 0] - qpos_before[..., 0]) / self._dt
        ctrl_cost = self.CTRL_COST_WEIGHT * (action**2).sum(dim=-1)

        z = self._dx.qpos[..., 2]
        soft_healthy = torch.sigmoid(10.0 * (z - self.HEALTHY_Z_LOW)) * torch.sigmoid(10.0 * (self.HEALTHY_Z_HIGH - z))
        # Multiplicative: forward velocity only counts while upright (dm_control style)
        reward = soft_healthy * (forward_vel + self.HEALTHY_REWARD) - ctrl_cost
        return reward.unsqueeze(-1).to(self.dtype)

    def _compute_terminated(self):
        return torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)


# ------------------------------------------------------------------
# Policy builder
# ------------------------------------------------------------------


def _make_policy(
    obs_dim: int,
    act_dim: int,
    device,
    use_batchnorm: bool = True,
    dropout_p: float = 0.1,
):
    layers = []
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(obs_dim, device=device))
    layers.append(
        MLP(
            in_features=obs_dim,
            out_features=act_dim,
            num_cells=[256, 256, 256, 256],
            activation_class=nn.ELU,
            dropout=dropout_p,
            device=device,
        ),
    )
    layers.append(nn.Tanh())
    net = nn.Sequential(*layers)
    return TensorDictModule(net, in_keys=["observation"], out_keys=["action"])


# ------------------------------------------------------------------
# Eval helper
# ------------------------------------------------------------------


def _run_eval(eval_env, policy, iteration, logger, max_steps=500):
    policy.eval()
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = eval_env.rollout(
            max_steps=max_steps,
            policy=policy,
            auto_reset=True,
            break_when_any_done=True,
        )
    ep_reward = rollout["next", "episode_reward"][:, -1].mean().item()
    step_rewards = rollout["next", "reward"].squeeze(-1)

    log_dict = {
        "eval/episode_reward": ep_reward,
        "eval/mean_step_reward": step_rewards.mean().item(),
        "iteration": iteration,
    }

    for t in eval_env.transform:
        if isinstance(t, VideoRecorder) and t.obs:
            try:
                import wandb

                vid = torch.stack(t.obs, 0).unsqueeze(0).cpu()
                log_dict["eval_video"] = wandb.Video(vid, fps=30, format="mp4")
            except Exception as e:
                mjt_logger.warning(f"  Video encoding failed: {e}")
            t.obs = []
            t.count = 0

    logger.experiment.log(log_dict)
    mjt_logger.info(f"  [EVAL] iter={iteration + 1} episode_reward={ep_reward:.2f}")
    policy.train()


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def train(args):
    device = args.device
    dtype = torch.float64

    train_env = TransformedEnv(
        SmoothHumanoidRichEnv(
            num_envs=args.num_envs,
            device=device,
            frame_skip=args.frame_skip,
        ),
        RewardSum(),
    )

    obs_dim = train_env.observation_spec["observation"].shape[-1]
    act_dim = train_env.action_spec.shape[-1]

    policy = _make_policy(
        obs_dim,
        act_dim,
        device,
        use_batchnorm=args.batchnorm,
        dropout_p=args.dropout,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_iters,
        eta_min=args.lr * 0.01,
    )

    logger = WandbLogger(
        exp_name="humanoid_rich_direct",
        project=args.wandb_project,
        config=vars(args),
    )

    # Eval env uses the rich obs variant too
    eval_env = TransformedEnv(
        HumanoidRichEnv(num_envs=1, device=device, frame_skip=args.frame_skip),
        Compose(
            PixelRenderTransform(out_keys=["pixels"]),
            VideoRecorder(
                logger=logger,
                tag="eval_video",
                skip=1,
                make_grid=False,
            ),
            RewardSum(),
        ),
    )

    mjt_logger.info(f"Direct backprop (rich obs) | obs={obs_dim} act={act_dim} device={device}")
    mjt_logger.info(
        f"  horizon={args.horizon} frame_skip={args.frame_skip} num_envs={args.num_envs} dt={train_env.base_env._dt}"
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    use_cuda = device is not None and "cuda" in str(device)

    for iteration in range(args.num_iters):
        td = train_env.reset()

        total_reward = torch.zeros(
            args.num_envs,
            1,
            device=device,
            dtype=dtype,
        )

        if use_cuda:
            torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()

        policy.train()
        with mujoco_torch.differentiable_mode(
            smooth_collisions=args.smooth_collisions,
            cfd=args.cfd,
            adaptive_integration=args.adaptive_integration,
        ):
            rollout = train_env.rollout(
                max_steps=args.horizon,
                policy=policy,
                auto_reset=False,
                break_when_any_done=False,
                tensordict=td,
            )
            total_reward = rollout["next", "episode_reward"][:, -1].unsqueeze(-1)

        loss = -total_reward.mean()

        if use_cuda:
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_fwd_start

        if use_cuda:
            torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            policy.parameters(),
            args.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if use_cuda:
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd_start

        mean_reward = total_reward.detach().mean().item()
        best_reward = max(best_reward, mean_reward)

        log_dict = {
            "train/mean_reward": mean_reward,
            "train/max_reward": total_reward.detach().max().item(),
            "train/best_mean_reward": best_reward,
            "train/loss": loss.detach().item(),
            "train/grad_norm": (grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "train/lr": scheduler.get_last_lr()[0],
            "perf/forward_s": t_fwd,
            "perf/backward_s": t_bwd,
            "iteration": iteration,
        }
        logger.experiment.log(log_dict)

        if (iteration + 1) % args.log_interval == 0 or iteration == 0:
            elapsed = time.perf_counter() - t0
            mjt_logger.info(
                f"  iter {iteration + 1}/{args.num_iters} | "
                f"reward={mean_reward:.2f} | best={best_reward:.2f} | "
                f"grad={grad_norm:.4f} | "
                f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s | "
                f"it/s={(iteration + 1) / elapsed:.2f}"
            )

        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            _run_eval(
                eval_env,
                policy,
                iteration,
                logger,
                args.max_eval_steps,
            )

    elapsed = time.perf_counter() - t0
    mjt_logger.info(f"Training done. {args.num_iters} iters in {elapsed:.1f}s. Best mean reward: {best_reward:.2f}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Direct backprop optimisation of Humanoid (rich obs)",
    )

    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_eval_steps", type=int, default=500)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batchnorm", action="store_true", default=True)
    parser.add_argument(
        "--no_batchnorm",
        dest="batchnorm",
        action="store_false",
    )

    parser.add_argument(
        "--smooth_collisions",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_smooth_collisions",
        dest="smooth_collisions",
        action="store_false",
    )
    parser.add_argument("--cfd", action="store_true", default=True)
    parser.add_argument("--no_cfd", dest="cfd", action="store_false")
    parser.add_argument(
        "--adaptive_integration",
        action="store_true",
        default=False,
    )

    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="mujoco-torch-zoo",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)


if __name__ == "__main__":
    main()
