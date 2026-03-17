#!/usr/bin/env python3
"""Train a PPO agent on mujoco-torch zoo environments.

Usage::

    python zoo/train_ppo.py --env halfcheetah
    python zoo/train_ppo.py --env cartpole --total_steps 200000
    python zoo/train_ppo.py --env ant --num_envs 32 --compile

"""

import argparse
import time

import torch
import torch.nn as nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, SamplerWithoutReplacement, TensorDictReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

from zoo import ENVS

# ------------------------------------------------------------------
# Network builders
# ------------------------------------------------------------------


def _make_actor(obs_dim: int, act_dim: int, device):
    """MLP policy that outputs TanhNormal distribution parameters."""
    net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=act_dim,
            num_cells=[256, 256],
            activation_class=nn.Tanh,
            device=device,
        ),
        AddStateIndependentNormalScale(act_dim, device=device),
    )
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
    return ProbabilisticActor(
        module=module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


def _make_critic(obs_dim: int, device):
    """MLP state-value function."""
    net = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=[256, 256],
        activation_class=nn.Tanh,
        device=device,
    )
    return ValueOperator(net, in_keys=["observation"])


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


def train_ppo(env, args, logger, eval_env=None):
    """On-policy PPO training loop using TorchRL collector + replay buffer."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    device = env.device

    actor = _make_actor(obs_dim, act_dim, device)
    critic = _make_critic(obs_dim, device)

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.01,
        critic_coeff=0.5,
        normalize_advantage=True,
    )
    loss_module.to(device)

    advantage_module = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=3e-4)

    frames_per_batch = args.num_envs * args.rollout_len
    mini_batch_size = frames_per_batch // args.num_minibatches

    collector = SyncDataCollector(
        env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=args.total_steps,
    )

    data_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=mini_batch_size,
    )

    num_batches = len(collector)
    print(f"PPO | obs_dim={obs_dim} act_dim={act_dim} device={device}")
    print(
        f"  frames_per_batch={frames_per_batch} "
        f"num_batches={num_batches} "
        f"mini_batch_size={mini_batch_size}"
    )

    total_frames = 0
    t0 = time.perf_counter()

    for batch_idx, batch in enumerate(collector):
        total_frames += batch.numel()

        for _epoch in range(args.ppo_epochs):
            with torch.no_grad():
                advantage_module(batch)
            data_buffer.extend(batch.reshape(-1))
            for mb in data_buffer:
                loss_vals = loss_module(mb)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 0.5)
                optimizer.step()

        collector.update_policy_weights_()

        fps = total_frames / (time.perf_counter() - t0)
        done_mask = batch["next", "done"].squeeze(-1)
        if done_mask.any():
            ep_reward = batch["next", "episode_reward"].squeeze(-1)[done_mask].mean().item()
        else:
            ep_reward = batch["next", "episode_reward"][:, -1].mean().item()
        logger.experiment.log({
            "reward/episode_reward": ep_reward,
            "perf/fps": fps,
            "global_step": total_frames,
        })

        if eval_env is not None and (batch_idx == 0 or (batch_idx + 1) % args.eval_interval == 0):
            _run_eval(eval_env, actor, total_frames, logger, max_steps=args.eval_steps)

        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == 0:
            print(
                f"  batch {batch_idx + 1}/{num_batches} | "
                f"frames={total_frames} | "
                f"fps={fps:.0f} | "
                f"mean_ep_reward={ep_reward:.2f}"
            )

    collector.shutdown()
    print(f"PPO training done. {total_frames} total frames.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train PPO on mujoco-torch zoo envs")
    parser.add_argument("--env", choices=list(ENVS), required=True, help="Environment name")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="torch.compile the physics step")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--rollout_len", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)

    parser.add_argument("--wandb_project", type=str, default="mujoco-torch-zoo")
    parser.add_argument("--record_video", action="store_true", help="Record eval videos to wandb")
    parser.add_argument("--eval_interval", type=int, default=10, help="Eval every N collector batches")
    parser.add_argument("--eval_steps", type=int, default=500, help="Max steps per eval episode")

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    logger = WandbLogger(
        exp_name=f"{args.env}_ppo",
        project=args.wandb_project,
        config=vars(args),
    )

    env_cls = ENVS[args.env]
    env_kwargs = {"device": args.device, "compile_step": args.compile}
    env = TransformedEnv(
        env_cls(num_envs=args.num_envs, **env_kwargs),
        Compose(RewardSum()),
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

    train_ppo(env, args, logger, eval_env=eval_env)


if __name__ == "__main__":
    main()
