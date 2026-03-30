#!/usr/bin/env python3
"""PPO training for MuJoCo locomotion via mujoco-torch + TorchRL.

Uses TorchRL collectors, loss modules, and transforms throughout.
The zoo envs use vmap internally for fast batched simulation.

Usage::

    python examples/train_ppo.py --env halfcheetah --num_envs 1024
    python examples/train_ppo.py --env humanoid --num_envs 512
    python examples/train_ppo.py --env ant --num_envs 1024
"""

import argparse
import time

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
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
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

from mujoco_torch.zoo import ENVS

# ------------------------------------------------------------------
# Environment factories
# ------------------------------------------------------------------


def make_env(
    env_name,
    num_envs,
    device,
    frame_skip,
    compile_step=False,
    compile_kwargs=None,
    obs_norm_num_iter=50,
):
    cls = ENVS[env_name]
    base = cls(
        num_envs=num_envs,
        device=device,
        frame_skip=frame_skip,
        compile_step=compile_step,
        compile_kwargs=compile_kwargs,
    )
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
    # Initialize obs_norm BEFORE adding PixelRenderTransform, which triggers
    # a reset during construction to discover the observation spec.
    if obs_norm_td is not None:
        obs_norm.loc = obs_norm_td["loc"].clone()
        obs_norm.scale = obs_norm_td["scale"].clone()
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
                fps=30,
            ),
        ),
    )
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
            activation_class=nn.Tanh,
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


def make_critic(obs_dim, device):
    return ValueOperator(
        module=MLP(
            in_features=obs_dim,
            out_features=1,
            num_cells=[256, 256],
            activation_class=nn.Tanh,
            device=device,
        ),
        in_keys=["observation"],
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
                t.iter = iteration
                t.dump()
            except Exception:
                pass

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
        exp_name=f"{args.env}_ppo",
        project=args.wandb_project,
        config=vars(args),
    )
    # Share obs normalization stats from train env to eval env
    obs_norm_td = train_env.transform[1].state_dict()
    print(f"  Train ObservationNorm state_dict keys={list(obs_norm_td.keys())}", flush=True)
    for k, v in obs_norm_td.items():
        print(f"    {k}: shape={v.shape} dtype={v.dtype} device={v.device}", flush=True)
    eval_env = make_eval_env(args.env, train_device, args.frame_skip, logger, obs_norm_td=obs_norm_td)

    # --- Models (on train device) ---
    actor = make_actor(obs_dim, act_dim, train_device)
    critic = make_critic(obs_dim, train_device)

    # --- GAE ---
    adv_module = GAE(
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        value_network=critic,
    )

    # --- PPO loss ---
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        entropy_coeff=args.entropy_coeff,
        critic_coeff=args.critic_coeff,
    )

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=args.lr,
        eps=1e-5,
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
        f"PPO [{args.env}] obs={obs_dim} act={act_dim}\n"
        f"  env_device={env_device} train_device={train_device}\n"
        f"  num_envs={args.num_envs} frames_per_batch={args.frames_per_batch} "
        f"total_frames={args.total_frames}\n"
        f"  gamma={args.gamma} gae_lambda={args.gae_lambda} "
        f"clip={args.clip_epsilon} epochs={args.num_epochs}",
        flush=True,
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    collected_frames = 0

    for iteration, batch in enumerate(collector):
        collected_frames += batch.numel()

        # --- GAE ---
        with torch.no_grad():
            adv_module(batch)
        adv = batch["advantage"]
        batch["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)

        # --- PPO epochs ---
        data = batch.reshape(-1)
        for _ in range(args.num_epochs):
            perm = torch.randperm(data.shape[0], device=train_device)
            for start in range(0, data.shape[0], args.mini_batch_size):
                idx = perm[start : start + args.mini_batch_size]
                mb = data[idx]
                loss_vals = loss_module(mb)
                total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    loss_module.parameters(),
                    args.max_grad_norm,
                )
                optim.step()

        # --- Logging ---
        ep_done = batch["next", "done"].squeeze(-1)
        ep_rews = batch["next", "episode_reward"][ep_done]
        mean_ep = ep_rews.mean().item() if ep_rews.numel() > 0 else float("nan")
        if mean_ep == mean_ep:  # not nan
            best_reward = max(best_reward, mean_ep)

        logger.experiment.log(
            {
                "train/mean_ep_reward": mean_ep,
                "train/best_ep_reward": best_reward,
                "train/mean_step_reward": batch["next", "reward"].mean().item(),
                "train/loss_objective": loss_vals["loss_objective"].item(),
                "train/loss_critic": loss_vals["loss_critic"].item(),
                "train/loss_entropy": loss_vals["loss_entropy"].item(),
                "collected_frames": collected_frames,
                "iteration": iteration,
            }
        )

        if (iteration + 1) % args.log_interval == 0 or iteration == 0:
            elapsed = time.perf_counter() - t0
            fps = collected_frames / elapsed
            print(
                f"  iter {iteration + 1} | frames={collected_frames} | "
                f"ep_reward={mean_ep:.1f} | best={best_reward:.1f} | "
                f"fps={fps:.0f}",
                flush=True,
            )

        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            eval_r = run_eval(eval_env, actor, iteration, logger)
            print(f"    [EVAL] ep_reward={eval_r:.1f}", flush=True)

    elapsed = time.perf_counter() - t0
    print(
        f"Done. {collected_frames} frames in {elapsed:.0f}s. Best ep reward: {best_reward:.1f}",
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="PPO for MuJoCo locomotion")

    # Env
    p.add_argument(
        "--env",
        type=str,
        default="halfcheetah",
        choices=list(ENVS.keys()),
    )
    p.add_argument("--num_envs", type=int, default=1024)
    p.add_argument("--frame_skip", type=int, default=5)

    # Collection
    p.add_argument("--frames_per_batch", type=int, default=65536)
    p.add_argument("--total_frames", type=int, default=10_000_000)

    # PPO
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    p.add_argument("--critic_coeff", type=float, default=0.5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--mini_batch_size", type=int, default=4096)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Logging
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--wandb_project", type=str, default="mujoco-torch-zoo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Env/collection device (default: cuda)")
    p.add_argument("--train_device", type=str, default=None, help="Training device (default: same as --device)")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument(
        "--compile_mode", type=str, default=None, help="torch.compile mode (e.g. 'reduce-overhead', 'max-autotune')"
    )

    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train(args)


if __name__ == "__main__":
    main()
