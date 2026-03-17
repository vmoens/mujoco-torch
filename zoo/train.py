#!/usr/bin/env python3
"""Train RL agents on mujoco-torch zoo environments.

Usage::

    python zoo/train.py --env halfcheetah --algo ppo
    python zoo/train.py --env ant --algo ddpg --num_envs 32
    python zoo/train.py --env cartpole --algo ppo --total_steps 200000

"""

import argparse
import time

import torch
import torch.nn as nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    MLP,
    OrnsteinUhlenbeckProcessModule,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss, DDPGLoss
from torchrl.objectives.value import GAE

from zoo import ENVS


# ------------------------------------------------------------------
# Network builders
# ------------------------------------------------------------------


def _make_ppo_actor(obs_dim: int, act_dim: int, device):
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
    actor = ProbabilisticActor(
        module=module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    return actor


def _make_ppo_critic(obs_dim: int, device):
    """MLP state-value function."""
    net = MLP(
        in_features=obs_dim,
        out_features=1,
        num_cells=[256, 256],
        activation_class=nn.Tanh,
        device=device,
    )
    return ValueOperator(net, in_keys=["observation"])


def _make_ddpg_actor(obs_dim: int, act_dim: int, device):
    """Deterministic MLP actor for DDPG (outputs tanh-squashed action)."""
    net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=act_dim,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
            activate_last_layer=False,
            device=device,
        ),
        nn.Tanh(),
    )
    return TensorDictModule(net, in_keys=["observation"], out_keys=["action"])


def _make_ddpg_critic(obs_dim: int, act_dim: int, device):
    """MLP Q-value function for DDPG (obs + action → scalar)."""
    net = MLP(
        in_features=obs_dim + act_dim,
        out_features=1,
        num_cells=[256, 256],
        activation_class=nn.ReLU,
        device=device,
    )

    class _QNet(nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def forward(self, observation, action):
            return self.mlp(torch.cat([observation, action], dim=-1))

    return TensorDictModule(
        _QNet(net),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )


# ------------------------------------------------------------------
# Training loops
# ------------------------------------------------------------------


def train_ppo(env, args):
    """On-policy PPO training loop."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    device = env.device

    actor = _make_ppo_actor(obs_dim, act_dim, device)
    critic = _make_ppo_critic(obs_dim, device)

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
    num_batches = args.total_steps // frames_per_batch
    mini_batch_size = frames_per_batch // args.num_minibatches

    print(f"PPO | obs_dim={obs_dim} act_dim={act_dim} device={device}")
    print(
        f"  frames_per_batch={frames_per_batch} "
        f"num_batches={num_batches} "
        f"mini_batch_size={mini_batch_size}"
    )

    total_frames = 0
    t0 = time.perf_counter()

    for batch_idx in range(num_batches):
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            rollout = env.rollout(
                max_steps=args.rollout_len,
                policy=actor,
                auto_reset=True,
                break_when_any_done=False,
            )

        total_frames += rollout.numel()

        with torch.no_grad():
            advantage_module(rollout)

        flat = rollout.reshape(-1)

        for _epoch in range(args.ppo_epochs):
            perm = torch.randperm(flat.shape[0], device=device)
            for start in range(0, flat.shape[0], mini_batch_size):
                idx = perm[start : start + mini_batch_size]
                mb = flat[idx]
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

        if (batch_idx + 1) % args.log_interval == 0 or batch_idx == 0:
            elapsed = time.perf_counter() - t0
            fps = total_frames / elapsed
            ep_reward = rollout["next", "reward"].sum(dim=-2).mean().item()
            print(
                f"  batch {batch_idx + 1}/{num_batches} | "
                f"frames={total_frames} | "
                f"fps={fps:.0f} | "
                f"mean_ep_reward={ep_reward:.2f}"
            )

    print(f"PPO training done. {total_frames} total frames.")


def train_ddpg(env, args):
    """Off-policy DDPG training loop."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]
    device = env.device

    actor = _make_ddpg_actor(obs_dim, act_dim, device)
    critic = _make_ddpg_critic(obs_dim, act_dim, device)

    loss_module = DDPGLoss(
        actor_network=actor,
        value_network=critic,
        loss_function="l2",
        delay_actor=True,
        delay_value=True,
    )
    loss_module.to(device)
    loss_module.make_value_estimator(gamma=0.99)

    target_updater = loss_module.target_value_network_params
    tau = 0.005

    exploration = OrnsteinUhlenbeckProcessModule(
        spec=env.action_spec,
        eps_init=1.0,
        eps_end=0.1,
        annealing_num_steps=args.total_steps // 2,
        device=device,
    )

    actor_opt = torch.optim.Adam(loss_module.actor_network_params.values(), lr=1e-4)
    critic_opt = torch.optim.Adam(loss_module.value_network_params.values(), lr=1e-3)

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device=device),
        batch_size=args.ddpg_batch_size,
    )

    print(f"DDPG | obs_dim={obs_dim} act_dim={act_dim} device={device}")

    total_frames = 0
    t0 = time.perf_counter()
    reward_log = []

    td = env.reset()

    for step_idx in range(args.total_steps // args.num_envs):
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = actor(td)
            td = exploration(td)

        next_td = env.step(td)
        replay_buffer.extend(next_td.reshape(-1))
        total_frames += args.num_envs

        reward_log.append(next_td["next", "reward"].mean().item())

        td = next_td["next"].clone()
        done_mask = td["done"].squeeze(-1)
        if done_mask.any():
            reset_td = env.reset(next_td["next"])
            td[done_mask] = reset_td[done_mask]

        if total_frames < args.ddpg_warmup:
            continue

        for _update in range(args.ddpg_updates_per_step):
            batch = replay_buffer.sample()
            loss_vals = loss_module(batch)

            critic_opt.zero_grad()
            loss_vals["loss_value"].backward()
            nn.utils.clip_grad_norm_(
                loss_module.value_network_params.values(), 1.0
            )
            critic_opt.step()

            actor_opt.zero_grad()
            loss_vals["loss_actor"].backward()
            nn.utils.clip_grad_norm_(
                loss_module.actor_network_params.values(), 1.0
            )
            actor_opt.step()

            # Soft-update target networks
            with torch.no_grad():
                for p_target, p in zip(
                    target_updater.values(True, True),
                    loss_module.actor_network_params.values(True, True),
                ):
                    p_target.data.lerp_(p.data, tau)
                for p_target, p in zip(
                    loss_module.target_value_network_params.values(True, True),
                    loss_module.value_network_params.values(True, True),
                ):
                    p_target.data.lerp_(p.data, tau)

        if (step_idx + 1) % args.log_interval == 0 or step_idx == 0:
            elapsed = time.perf_counter() - t0
            fps = total_frames / elapsed
            recent = reward_log[-args.log_interval :]
            mean_r = sum(recent) / len(recent)
            print(
                f"  step {step_idx + 1} | "
                f"frames={total_frames} | "
                f"fps={fps:.0f} | "
                f"mean_reward={mean_r:.4f}"
            )

    print(f"DDPG training done. {total_frames} total frames.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train RL on mujoco-torch zoo envs")
    parser.add_argument(
        "--env",
        choices=list(ENVS),
        required=True,
        help="Environment name",
    )
    parser.add_argument(
        "--algo",
        choices=["ppo", "ddpg"],
        default="ppo",
        help="Algorithm (default: ppo)",
    )
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--total_steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=10)

    # PPO-specific
    parser.add_argument("--rollout_len", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)

    # DDPG-specific
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--ddpg_batch_size", type=int, default=256)
    parser.add_argument("--ddpg_warmup", type=int, default=10_000)
    parser.add_argument("--ddpg_updates_per_step", type=int, default=1)

    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)

    env_cls = ENVS[args.env]
    env = env_cls(num_envs=args.num_envs, device=args.device)
    print(f"Env: {args.env} | batch_size={env.batch_size} | device={env.device}")

    if args.algo == "ppo":
        train_ppo(env, args)
    else:
        train_ddpg(env, args)


if __name__ == "__main__":
    main()
