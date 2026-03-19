#!/usr/bin/env python3
"""Train a SAC agent on mujoco-torch zoo environments.

Usage::

    python -m mujoco_torch.zoo.train_sac --env halfcheetah
    python -m mujoco_torch.zoo.train_sac --env cartpole --total_steps 100000
    python -m mujoco_torch.zoo.train_sac --env ant --num_envs 64 --compile

"""

import argparse
import time

import torch
import torch.nn as nn
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, InitTracker, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import SoftUpdate, group_optimizers
from torchrl.objectives.sac import SACLoss
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

from mujoco_torch.zoo import ENVS

# ------------------------------------------------------------------
# Network builders
# ------------------------------------------------------------------


def _make_actor(obs_dim: int, act_dim: int, device):
    """Stochastic MLP actor for SAC (TanhNormal policy)."""
    net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
            device=device,
        ),
        NormalParamExtractor(),
    )
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
    return ProbabilisticActor(
        module=module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )


def _make_critic(obs_dim: int, act_dim: int, device):
    """MLP Q-value function for SAC (obs + action -> scalar)."""
    net = MLP(
        in_features=obs_dim + act_dim,
        out_features=1,
        num_cells=[256, 256],
        activation_class=nn.ReLU,
        device=device,
    )
    return ValueOperator(module=net, in_keys=["action", "observation"])


# ------------------------------------------------------------------
# Eval helper
# ------------------------------------------------------------------


def _run_eval(eval_env, actor, total_frames, logger, max_steps=200):
    """Run one eval episode, log reward and dump video (if recording)."""
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = eval_env.rollout(
            max_steps=max_steps,
            policy=actor,
            auto_reset=True,
            break_when_any_done=True,
        )
    ep_reward = rollout["next", "episode_reward"][:, -1].mean().item()
    log_dict = {
        "eval/episode_reward": ep_reward,
        "global_step": total_frames,
    }
    for t in eval_env.transform:
        if isinstance(t, VideoRecorder) and t.obs:
            import wandb

            vid = torch.stack(t.obs, 0).unsqueeze(0).cpu()
            log_dict["eval_video"] = wandb.Video(vid, fps=30, format="mp4")
            t.obs = []
            t.count = 0
    logger.experiment.log(log_dict)
    print(f"  [EVAL] frames={total_frames} episode_reward={ep_reward:.2f}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def train_sac(env, args, logger, eval_env=None):
    """Off-policy SAC training loop following TorchRL SOTA pattern."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    device = env.device

    actor = _make_actor(obs_dim, act_dim, device)
    critic = _make_critic(obs_dim, act_dim, device)

    loss_module = SACLoss(
        actor_network=actor,
        qvalue_network=critic,
        num_qvalue_nets=2,
        loss_function="l2",
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=1.0,
        action_spec=env.action_spec_unbatched,
    )
    loss_module.to(device)
    loss_module.make_value_estimator(gamma=0.99)

    target_updater = SoftUpdate(loss_module, tau=0.005)

    optimizer_actor = torch.optim.Adam(
        loss_module.actor_network_params.flatten_keys().values(),
        lr=3e-4,
    )
    optimizer_critic = torch.optim.Adam(
        loss_module.qvalue_network_params.flatten_keys().values(),
        lr=3e-4,
    )
    optimizer_alpha = torch.optim.Adam([loss_module.log_alpha], lr=3e-4)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    storage_kwargs = {"max_size": args.buffer_size}
    if device is not None:
        storage_kwargs["device"] = device
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(**storage_kwargs),
        batch_size=args.batch_size,
    )

    collector = SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=args.num_envs,
        total_frames=args.total_steps,
        init_random_frames=args.warmup,
    )

    num_updates = max(1, int(args.num_envs * args.utd_ratio))

    print(f"SAC | obs_dim={obs_dim} act_dim={act_dim} device={device}")
    print(f"  utd_ratio={args.utd_ratio} num_updates={num_updates}")

    total_frames = 0
    t0 = time.perf_counter()
    reward_log = []

    for step_idx, batch in enumerate(collector):
        replay_buffer.extend(batch.reshape(-1))
        total_frames += batch.numel()

        reward_log.append(batch["next", "reward"].mean().item())

        if total_frames >= args.warmup:
            for _ in range(num_updates):
                sample = replay_buffer.sample()
                loss_td = loss_module(sample)

                total_loss = loss_td["loss_actor"] + loss_td["loss_qvalue"] + loss_td["loss_alpha"]
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                target_updater.step()

            collector.update_policy_weights_()

        step_reward = batch["next", "reward"].mean().item()
        fps = total_frames / (time.perf_counter() - t0)
        log_dict = {
            "reward/step_mean": step_reward,
            "perf/fps": fps,
            "global_step": total_frames,
        }
        done_mask = batch["next", "done"].squeeze(-1)
        if done_mask.any():
            ep_reward = batch["next", "episode_reward"].squeeze(-1)[done_mask].mean().item()
            log_dict["reward/episode_reward"] = ep_reward
        if total_frames >= args.warmup:
            log_dict["train/alpha"] = loss_td["alpha"].item()
            log_dict["train/entropy"] = loss_td["entropy"].item()
        logger.experiment.log(log_dict)

        if eval_env is not None and (step_idx == 0 or (step_idx + 1) % args.eval_interval == 0):
            _run_eval(eval_env, actor, total_frames, logger, max_steps=args.eval_steps)

        if (step_idx + 1) % args.log_interval == 0 or step_idx == 0:
            recent = reward_log[-args.log_interval :]
            mean_r = sum(recent) / len(recent)
            print(f"  step {step_idx + 1} | frames={total_frames} | fps={fps:.0f} | mean_reward={mean_r:.4f}")

    collector.shutdown()
    print(f"SAC training done. {total_frames} total frames.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train SAC on mujoco-torch zoo envs")
    parser.add_argument("--env", choices=list(ENVS), required=True, help="Environment name")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the physics step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument(
        "--utd_ratio", type=float, default=0.25, help="Update-to-data ratio (gradient updates per env step)"
    )

    parser.add_argument("--wandb_project", type=str, default="mujoco-torch-zoo")
    parser.add_argument("--record_video", action="store_true", help="Record eval videos to wandb")
    parser.add_argument("--eval_interval", type=int, default=10, help="Eval every N collector steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Max steps per eval episode")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    logger = WandbLogger(
        exp_name=f"{args.env}_sac",
        project=args.wandb_project,
        config=vars(args),
    )

    env_cls = ENVS[args.env]
    env_kwargs = {"device": args.device, "compile_step": args.compile}
    env = TransformedEnv(
        env_cls(num_envs=args.num_envs, **env_kwargs),
        Compose(InitTracker(), RewardSum()),
    )
    print(f"Env: {args.env} | batch_size={env.batch_size} | device={env.device}")

    eval_transforms = [RewardSum()]
    if args.record_video:
        eval_transforms = [
            PixelRenderTransform(out_keys=["pixels"]),
            VideoRecorder(logger=logger, tag="eval_video", skip=1, make_grid=False),
        ] + eval_transforms
    eval_env = TransformedEnv(
        env_cls(num_envs=1, **env_kwargs),
        Compose(*eval_transforms),
    )

    train_sac(env, args, logger, eval_env=eval_env)


if __name__ == "__main__":
    main()
