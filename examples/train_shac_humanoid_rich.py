#!/usr/bin/env python3
"""SHAC training for Humanoid with rich observations.

Combines short-horizon differentiable rollouts with a learned value function
(terminal bootstrap) and stochastic policy (exploration via entropy).

Usage::

    python examples/train_shac_humanoid_rich.py
    python examples/train_shac_humanoid_rich.py --device cuda --num_envs 128
"""

import argparse
import sys
import time
from pathlib import Path

# Allow importing sibling example modules when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    NormalParamExtractor,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

import mujoco_torch
from mujoco_torch._src.log import logger as mjt_logger
from mujoco_torch.zoo.humanoid_rich import HumanoidRichEnv

from shac_loss import SHACLoss
from train_direct_humanoid_rich import SmoothHumanoidRichEnv


# ------------------------------------------------------------------
# Network builders
# ------------------------------------------------------------------


def _make_actor(obs_dim: int, act_dim: int, device, use_batchnorm: bool = True):
    layers = []
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(obs_dim, device=device))
    layers.append(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=[256, 256],
            activation_class=nn.ELU,
            device=device,
        ),
    )
    layers.append(NormalParamExtractor())
    actor_net = nn.Sequential(*layers)

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


def _make_critic(obs_dim: int, device, use_batchnorm: bool = True):
    layers = []
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(obs_dim, device=device))
    layers.append(
        MLP(
            in_features=obs_dim,
            out_features=1,
            num_cells=[256, 256],
            activation_class=nn.ELU,
            device=device,
        ),
    )
    critic_net = nn.Sequential(*layers)
    return ValueOperator(
        module=critic_net,
        in_keys=["observation"],
    )


# ------------------------------------------------------------------
# Eval helper (adapted from train_direct_humanoid_rich.py)
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
    mjt_logger.info(
        f"  [EVAL] iter={iteration + 1} episode_reward={ep_reward:.2f}"
    )
    policy.train()


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def train(args):
    device = args.device
    dtype = torch.float64

    train_env = SmoothHumanoidRichEnv(
        num_envs=args.num_envs,
        device=device,
        frame_skip=args.frame_skip,
    )

    obs_dim = train_env.observation_spec["observation"].shape[-1]
    act_dim = train_env.action_spec.shape[-1]

    actor = _make_actor(obs_dim, act_dim, device, use_batchnorm=args.batchnorm)
    critic = _make_critic(obs_dim, device, use_batchnorm=args.batchnorm)

    shac = SHACLoss(
        actor_network=actor,
        value_network=critic,
        gamma=args.gamma,
        tau=args.tau,
        act_dim=act_dim,
    ).to(device)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
    alpha_optim = torch.optim.Adam([shac.log_alpha], lr=args.lr_actor)

    logger = WandbLogger(
        exp_name="humanoid_rich_shac",
        project=args.wandb_project,
        config=vars(args),
    )

    eval_env = TransformedEnv(
        HumanoidRichEnv(num_envs=1, device=device, frame_skip=args.frame_skip),
        Compose(
            PixelRenderTransform(out_keys=["pixels"]),
            VideoRecorder(
                logger=logger, tag="eval_video", skip=1, make_grid=False,
            ),
            RewardSum(),
        ),
    )

    mjt_logger.info(
        f"SHAC (rich obs) | obs={obs_dim} act={act_dim} device={device}"
    )
    mjt_logger.info(
        f"  horizon={args.horizon} frame_skip={args.frame_skip} "
        f"num_envs={args.num_envs} gamma={args.gamma} tau={args.tau}"
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    use_cuda = device is not None and "cuda" in str(device)

    for iteration in range(args.num_iters):
        td = train_env.reset()
        rollout_tds = []

        if use_cuda:
            torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()

        actor.train()
        critic.train()

        with mujoco_torch.differentiable_mode(
            smooth_collisions=args.smooth_collisions,
            cfd=args.cfd,
            adaptive_integration=args.adaptive_integration,
        ):
            for _t in range(args.horizon):
                td = actor(td)
                next_td = train_env.step(td)
                # Store step data for loss computation
                step_td = TensorDict(
                    {
                        "observation": td["observation"],
                        "action": td["action"],
                        "sample_log_prob": td["action_log_prob"],
                        "next": TensorDict(
                            {
                                "observation": next_td["next", "observation"],
                                "reward": next_td["next", "reward"],
                            },
                            batch_size=td.batch_size,
                        ),
                    },
                    batch_size=td.batch_size,
                )
                rollout_tds.append(step_td)
                td = next_td["next"]

        rollout = torch.stack(rollout_tds, 0)  # [T, B, ...]

        if use_cuda:
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t_fwd_start

        # ----------------------------------------------------------
        # Actor + alpha update (backprops through physics)
        # ----------------------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        t_bwd_start = time.perf_counter()

        actor_optim.zero_grad()
        alpha_optim.zero_grad()
        loss_actor, mean_lp = shac.actor_loss(rollout)
        loss_alpha = shac.alpha_loss(mean_lp)
        (loss_actor + loss_alpha).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            actor.parameters(), args.grad_clip,
        )
        actor_optim.step()
        alpha_optim.step()

        # ----------------------------------------------------------
        # Critic update (no physics gradients, cheap)
        # ----------------------------------------------------------
        critic_optim.zero_grad()
        loss_critic = shac.critic_loss(rollout.detach())
        loss_critic.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            critic.parameters(), args.grad_clip,
        )
        critic_optim.step()

        shac.update_target()

        if use_cuda:
            torch.cuda.synchronize()
        t_bwd = time.perf_counter() - t_bwd_start

        # ----------------------------------------------------------
        # Logging
        # ----------------------------------------------------------
        total_reward = rollout["next", "reward"].sum(0).detach()  # [B, 1]
        mean_reward = total_reward.mean().item()
        best_reward = max(best_reward, mean_reward)

        log_dict = {
            "train/mean_reward": mean_reward,
            "train/max_reward": total_reward.max().item(),
            "train/best_mean_reward": best_reward,
            "train/loss_actor": loss_actor.detach().item(),
            "train/loss_critic": loss_critic.detach().item(),
            "train/loss_alpha": loss_alpha.detach().item(),
            "train/alpha": shac.alpha.detach().item(),
            "train/mean_log_prob": mean_lp.item(),
            "train/actor_grad_norm": (
                actor_grad_norm.item()
                if isinstance(actor_grad_norm, torch.Tensor)
                else actor_grad_norm
            ),
            "train/critic_grad_norm": (
                critic_grad_norm.item()
                if isinstance(critic_grad_norm, torch.Tensor)
                else critic_grad_norm
            ),
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
                f"alpha={shac.alpha.item():.4f} | "
                f"L_a={loss_actor.item():.4f} L_c={loss_critic.item():.4f} | "
                f"fwd={t_fwd:.2f}s bwd={t_bwd:.2f}s | "
                f"it/s={(iteration + 1) / elapsed:.2f}"
            )

        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            _run_eval(
                eval_env, actor, iteration, logger, args.max_eval_steps,
            )

    elapsed = time.perf_counter() - t0
    mjt_logger.info(
        f"Training done. {args.num_iters} iters in {elapsed:.1f}s. "
        f"Best mean reward: {best_reward:.2f}"
    )


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SHAC training of Humanoid (rich obs)",
    )

    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_eval_steps", type=int, default=500)

    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--batchnorm", action="store_true", default=True)
    parser.add_argument(
        "--no_batchnorm", dest="batchnorm", action="store_false",
    )

    parser.add_argument(
        "--smooth_collisions", action="store_true", default=True,
    )
    parser.add_argument(
        "--no_smooth_collisions",
        dest="smooth_collisions",
        action="store_false",
    )
    parser.add_argument("--cfd", action="store_true", default=True)
    parser.add_argument("--no_cfd", dest="cfd", action="store_false")
    parser.add_argument(
        "--adaptive_integration", action="store_true", default=False,
    )

    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--wandb_project", type=str, default="mujoco-torch-zoo",
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
