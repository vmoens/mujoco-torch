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

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    RewardSum,
    StepCounter,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import SACLoss
from torchrl.objectives.utils import SoftUpdate
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

from mujoco_torch.zoo import ENVS


# ------------------------------------------------------------------
# Environment factories
# ------------------------------------------------------------------


def make_env(env_name, num_envs, device, frame_skip):
    cls = ENVS[env_name]
    base = cls(num_envs=num_envs, device=device, frame_skip=frame_skip)
    return TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            StepCounter(max_steps=1000),
            RewardSum(),
        ),
    )


def make_eval_env(env_name, device, frame_skip, logger):
    cls = ENVS[env_name]
    base = cls(num_envs=1, device=device, frame_skip=frame_skip)
    return TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            StepCounter(max_steps=1000),
            RewardSum(),
            PixelRenderTransform(out_keys=["pixels"]),
            VideoRecorder(
                logger=logger, tag="eval_video", skip=1, make_grid=False,
            ),
        ),
    )


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
                    vid, fps=30, format="mp4",
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
    device = args.device

    # --- Envs ---
    train_env = make_env(
        args.env, args.num_envs, device, args.frame_skip,
    )
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    act_dim = train_env.action_spec.shape[-1]

    # --- Logger ---
    logger = WandbLogger(
        exp_name=f"{args.env}_sac",
        project=args.wandb_project,
        config=vars(args),
    )
    eval_env = make_eval_env(args.env, device, args.frame_skip, logger)

    # --- Models ---
    actor = make_actor(obs_dim, act_dim, device)
    qvalue = make_qvalue(obs_dim, act_dim, device)

    # --- SAC loss ---
    # target_entropy = -act_dim (standard heuristic)
    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=qvalue,
        num_qvalue_nets=2,
        action_spec=train_env.action_spec,
    )
    loss_module.make_value_estimator(
        loss_module.default_value_estimator, gamma=args.gamma,
    )

    # Target network soft-update
    target_updater = SoftUpdate(loss_module, tau=args.tau)

    # Separate optimizers (standard for SAC)
    actor_optim = torch.optim.Adam(
        list(loss_module.actor_network_params.values(True, True)), lr=args.lr,
    )
    critic_optim = torch.optim.Adam(
        list(loss_module.qvalue_network_params.values(True, True)), lr=args.lr,
    )
    alpha_optim = torch.optim.Adam(
        [loss_module.log_alpha], lr=args.lr,
    )

    # --- Replay buffer ---
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size, device=device),
        batch_size=args.batch_size,
    )

    # --- Collector ---
    collector = SyncDataCollector(
        train_env,
        actor,
        frames_per_batch=args.num_envs,  # one step across all envs
        total_frames=args.total_frames,
        device=device,
    )

    print(
        f"SAC [{args.env}] obs={obs_dim} act={act_dim} device={device}\n"
        f"  num_envs={args.num_envs} buffer_size={args.buffer_size} "
        f"batch_size={args.batch_size}\n"
        f"  gamma={args.gamma} tau={args.tau} lr={args.lr}",
        flush=True,
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    collected_frames = 0
    num_updates = 0

    for collect_iter, batch in enumerate(collector):
        collected_frames += batch.numel()

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
        for _ in range(args.utd_ratio):
            sample = buffer.sample()
            loss_vals = loss_module(sample)

            # Zero all grads
            critic_optim.zero_grad()
            actor_optim.zero_grad()
            alpha_optim.zero_grad()

            # Single combined backward
            total_loss = (
                loss_vals["loss_qvalue"]
                + loss_vals["loss_actor"]
                + loss_vals["loss_alpha"]
            )
            total_loss.backward()

            # Clip critic grads
            nn.utils.clip_grad_norm_(
                list(loss_module.qvalue_network_params.values(True, True)), 1.0,
            )

            # Step all optimizers
            critic_optim.step()
            actor_optim.step()
            alpha_optim.step()

            # Target update
            target_updater.step()
            num_updates += 1

        # --- Logging ---
        if (collect_iter + 1) % args.log_interval == 0:
            ep_done = batch["next", "done"].squeeze(-1)
            ep_rews = batch["next", "episode_reward"][ep_done]
            mean_ep = (
                ep_rews.mean().item() if ep_rews.numel() > 0 else float("nan")
            )
            if mean_ep == mean_ep:
                best_reward = max(best_reward, mean_ep)

            alpha = loss_module.log_alpha.exp().item()

            logger.experiment.log({
                "train/mean_ep_reward": mean_ep,
                "train/best_ep_reward": best_reward,
                "train/loss_actor": loss_vals["loss_actor"].item(),
                "train/loss_qvalue": loss_vals["loss_qvalue"].item(),
                "train/loss_alpha": loss_vals["loss_alpha"].item(),
                "train/alpha": alpha,
                "train/num_updates": num_updates,
                "collected_frames": collected_frames,
                "iteration": collect_iter,
            })

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
        f"Done. {collected_frames} frames in {elapsed:.0f}s. "
        f"Best ep reward: {best_reward:.1f}",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="SAC for MuJoCo locomotion")

    # Env
    p.add_argument(
        "--env", type=str, default="halfcheetah", choices=list(ENVS.keys()),
    )
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--frame_skip", type=int, default=5)

    # Collection
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
    p.add_argument("--device", type=str, default=None)

    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)


if __name__ == "__main__":
    main()
