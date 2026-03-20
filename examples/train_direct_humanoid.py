#!/usr/bin/env python3
"""Direct backprop optimization of Humanoid via differentiable mujoco-torch.

Instead of model-free RL (PPO/SAC), this script directly backpropagates through
the differentiable physics simulator to optimize a policy network.  Uses
``mujoco_torch.differentiable_mode()`` for smooth gradient flow through contacts.

The policy is unrolled for a fixed horizon inside the differentiable physics,
the total reward is computed, and gradients flow back through the entire
trajectory to update the policy weights via Adam.

Features:
    - Direct trajectory optimization through differentiable MuJoCo physics
    - Batched simulation via ``torch.vmap``
    - TorchRL ``TensorDictModule`` policy (compatible with eval env)
    - TorchRL transforms for evaluation (``RewardSum``, ``VideoRecorder``)
    - WandB logging with eval videos, reward curves

Usage::

    python examples/train_direct_humanoid.py
    python examples/train_direct_humanoid.py --device cuda --num_envs 128
    python examples/train_direct_humanoid.py --horizon 100 --no_cfd
"""

import argparse
import time

import mujoco
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import Compose, RewardSum
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP
from torchrl.record import PixelRenderTransform, VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger

import mujoco_torch
from mujoco_torch._src.log import logger as mjt_logger
from mujoco_torch.zoo import ENVS

# ------------------------------------------------------------------
# Network builder
# ------------------------------------------------------------------


def _make_policy(obs_dim: int, act_dim: int, device):
    """Deterministic MLP policy: observation -> tanh-squashed action."""
    net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=act_dim,
            num_cells=[256, 256],
            activation_class=nn.ELU,
            device=device,
        ),
        nn.Tanh(),
    )
    return TensorDictModule(net, in_keys=["observation"], out_keys=["action"])


# ------------------------------------------------------------------
# Differentiable simulation helpers
# ------------------------------------------------------------------


def _make_obs(dx, dtype):
    """Build observation from sim data: qpos[2:] + clipped qvel."""
    qpos = dx.qpos.to(dtype)
    qvel = dx.qvel.to(dtype).clamp(-10.0, 10.0)
    return torch.cat([qpos[..., 2:], qvel], dim=-1)


def _compute_reward(dx, qpos_before, action, dt, args):
    """Compute reward with smooth healthy bonus for gradient flow.

    Uses sigmoid soft-indicators instead of hard thresholds so that
    gradients propagate through the healthy/unhealthy boundary.
    """
    forward_vel = (dx.qpos[..., 0] - qpos_before[..., 0]) / dt

    # Smooth healthy reward (sigmoid approximation of indicator)
    z = dx.qpos[..., 2]
    soft_healthy = torch.sigmoid(10.0 * (z - args.healthy_z_low)) * torch.sigmoid(
        10.0 * (args.healthy_z_high - z)
    )
    healthy_reward = args.healthy_reward * soft_healthy

    ctrl_cost = args.ctrl_cost_weight * (action**2).sum(dim=-1)

    return forward_vel + healthy_reward - ctrl_cost


def _make_initial_states(mx, m_mj, num_envs, device, noise_scale=0.01):
    """Create a batch of initial simulation states with noise."""
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)
    dx0 = mujoco_torch.device_put(d_mj)
    if device is not None:
        dx0 = dx0.to(device)
    # Run one step so all dtypes match what vmap(step) produces.
    dx0 = mujoco_torch.step(mx, dx0)

    batch = torch.stack([dx0.clone() for _ in range(num_envs)])
    if noise_scale > 0:
        batch.qpos = batch.qpos + torch.empty_like(batch.qpos).uniform_(
            -noise_scale, noise_scale
        )
        batch.qvel = batch.qvel + torch.empty_like(batch.qvel).uniform_(
            -noise_scale, noise_scale
        )
    return batch


# ------------------------------------------------------------------
# Eval helper
# ------------------------------------------------------------------


def _run_eval(eval_env, policy, iteration, logger, max_steps=500):
    """Run one eval episode, log reward and dump video."""
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
                log_dict["eval_video"] = wandb.Video(
                    vid, fps=30, format="mp4"
                )
            except Exception as e:
                mjt_logger.warning(f"  Video encoding failed: {e}")
            t.obs = []
            t.count = 0

    logger.experiment.log(log_dict)
    mjt_logger.info(f"  [EVAL] iter={iteration + 1} episode_reward={ep_reward:.2f}")


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def train_direct(args):
    """Direct backprop training loop through differentiable physics."""
    device = args.device
    dtype = torch.float64

    # Load MuJoCo model
    from pathlib import Path

    xml_path = (
        Path(__file__).resolve().parent.parent
        / "mujoco_torch"
        / "test_data"
        / "humanoid.xml"
    )
    m_mj = mujoco.MjModel.from_xml_path(str(xml_path))
    mx = mujoco_torch.device_put(m_mj)
    if device is not None:
        mx = mx.to(device)
    dt = m_mj.opt.timestep * args.frame_skip

    # Dimensions
    obs_dim = 53  # qpos[2:](26) + qvel(27)
    act_dim = m_mj.nu  # 21

    # Policy
    policy = _make_policy(obs_dim, act_dim, device)

    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_iters, eta_min=args.lr * 0.01
    )

    # Physics step (with fixed_iterations for differentiability)
    def physics_step(d):
        return mujoco_torch.step(mx, d, fixed_iterations=True)

    # WandB logger
    logger = WandbLogger(
        exp_name="humanoid_direct_backprop",
        project=args.wandb_project,
        config=vars(args),
    )

    # Eval env (zoo HumanoidEnv + TorchRL transforms + video recording)
    eval_env_cls = ENVS["humanoid"]
    eval_transforms = [
        PixelRenderTransform(out_keys=["pixels"]),
        VideoRecorder(
            logger=logger, tag="eval_video", skip=1, make_grid=False
        ),
        RewardSum(),
    ]
    eval_env = TransformedEnv(
        eval_env_cls(num_envs=1, device=device, frame_skip=args.frame_skip),
        Compose(*eval_transforms),
    )

    mjt_logger.info(
        f"Direct backprop | obs={obs_dim} act={act_dim} device={device}"
    )
    mjt_logger.info(
        f"  horizon={args.horizon} frame_skip={args.frame_skip} "
        f"num_envs={args.num_envs} dt={dt}"
    )
    mjt_logger.info(
        f"  diff_mode: smooth={args.smooth_collisions} "
        f"cfd={args.cfd} adaptive={args.adaptive_integration}"
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")

    for iteration in range(args.num_iters):
        # Fresh batch of initial states each iteration
        dx = _make_initial_states(mx, m_mj, args.num_envs, device)

        total_reward = torch.zeros(args.num_envs, device=device, dtype=dtype)
        step_rewards = []

        # Unroll trajectory with differentiable physics
        with mujoco_torch.differentiable_mode(
            smooth_collisions=args.smooth_collisions,
            cfd=args.cfd,
            adaptive_integration=args.adaptive_integration,
        ):
            for _t in range(args.horizon):
                obs = _make_obs(dx, dtype)

                td = TensorDict(
                    {"observation": obs}, batch_size=[args.num_envs]
                )
                td = policy(td)
                action = td["action"]

                qpos_before = dx.qpos.clone()

                dx = dx.replace(ctrl=action)
                for _ in range(args.frame_skip):
                    dx = torch.vmap(physics_step)(dx)

                reward = _compute_reward(dx, qpos_before, action, dt, args)
                total_reward = total_reward + reward
                step_rewards.append(reward.detach().mean().item())

        # Loss = negative mean return
        loss = -total_reward.mean()

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            policy.parameters(), args.grad_clip
        )
        optimizer.step()
        scheduler.step()

        # Logging
        mean_reward = total_reward.detach().mean().item()
        max_reward_iter = total_reward.detach().max().item()
        best_reward = max(best_reward, mean_reward)

        log_dict = {
            "train/mean_reward": mean_reward,
            "train/max_reward": max_reward_iter,
            "train/best_mean_reward": best_reward,
            "train/loss": loss.detach().item(),
            "train/grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm,
            "train/lr": scheduler.get_last_lr()[0],
            "train/mean_step_reward": sum(step_rewards) / len(step_rewards),
            "iteration": iteration,
        }
        logger.experiment.log(log_dict)

        if (iteration + 1) % args.log_interval == 0 or iteration == 0:
            elapsed = time.perf_counter() - t0
            iters_per_sec = (iteration + 1) / elapsed
            mjt_logger.info(
                f"  iter {iteration + 1}/{args.num_iters} | "
                f"reward={mean_reward:.2f} | "
                f"best={best_reward:.2f} | "
                f"grad={grad_norm:.4f} | "
                f"it/s={iters_per_sec:.2f} | "
                f"time={elapsed:.1f}s"
            )

        # Eval with video
        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            _run_eval(
                eval_env, policy, iteration, logger, args.max_eval_steps
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
        description="Direct backprop optimization of Humanoid"
    )

    # Env
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--frame_skip", type=int, default=2)
    parser.add_argument("--max_eval_steps", type=int, default=500)

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Differentiable mode
    parser.add_argument(
        "--smooth_collisions", action="store_true", default=True
    )
    parser.add_argument(
        "--no_smooth_collisions",
        dest="smooth_collisions",
        action="store_false",
    )
    parser.add_argument("--cfd", action="store_true", default=True)
    parser.add_argument("--no_cfd", dest="cfd", action="store_false")
    parser.add_argument(
        "--adaptive_integration", action="store_true", default=False
    )

    # Reward
    parser.add_argument("--healthy_reward", type=float, default=5.0)
    parser.add_argument("--ctrl_cost_weight", type=float, default=0.1)
    parser.add_argument("--healthy_z_low", type=float, default=1.0)
    parser.add_argument("--healthy_z_high", type=float, default=2.0)

    # Logging
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument(
        "--wandb_project", type=str, default="mujoco-torch-zoo"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    train_direct(args)


if __name__ == "__main__":
    main()
