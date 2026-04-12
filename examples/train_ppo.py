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
import logging
import os
import time
from functools import partial

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Evaluator, MultiaSyncDataCollector, SyncDataCollector, aSyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, SamplerWithoutReplacement
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    RewardSum,
    StepCounter,
)
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

from mujoco_torch._src.log import logger as mjt_logger
from mujoco_torch.zoo import ENVS

if os.environ.get("MUJOCO_TORCH_TORCHRL_DEBUG"):
    torchrl_logger.setLevel(logging.DEBUG)

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
    obs_norm_td=None,
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
    if obs_norm_td is not None:
        env.transform[1].loc = obs_norm_td["loc"].to(device).clone()
        env.transform[1].scale = obs_norm_td["scale"].to(device).clone()
    return env


def init_obs_norm_stats(env, num_iter):
    mjt_logger.info(f"  ObservationNorm: computing stats ({num_iter} rollouts, {env.batch_size[0]} envs)...")
    t_init = time.perf_counter()
    env.transform[1].init_stats(num_iter, reduce_dim=(0, 1), cat_dim=0)
    obs_norm = env.transform[1]
    elapsed = time.perf_counter() - t_init
    mjt_logger.info(
        f"  ObservationNorm: done in {elapsed:.1f}s"
        f"  loc=[{obs_norm.loc.min():.2f}, {obs_norm.loc.max():.2f}]"
        f"  scale=[{obs_norm.scale.min():.2f}, {obs_norm.scale.max():.2f}]"
    )
    return obs_norm.state_dict()


def make_eval_env(
    env_name,
    device,
    frame_skip,
    logger,
    obs_norm_td=None,
    compile_step=False,
    compile_kwargs=None,
):
    cls = ENVS[env_name]
    base = cls(
        num_envs=1,
        device=device,
        frame_skip=frame_skip,
        compile_step=compile_step,
        compile_kwargs=compile_kwargs,
    )
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    # Initialize obs_norm BEFORE adding PixelRenderTransform, which triggers
    # a reset during construction to discover the observation spec.
    if obs_norm_td is not None:
        obs_norm.loc = obs_norm_td["loc"].to(device).clone()
        obs_norm.scale = obs_norm_td["scale"].to(device).clone()
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


def make_actor(obs_dim, act_dim, device, hidden_size=256, num_layers=4):
    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=[hidden_size] * num_layers,
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


def make_critic(obs_dim, device, hidden_size=256, num_layers=4):
    return ValueOperator(
        module=MLP(
            in_features=obs_dim,
            out_features=1,
            num_cells=[hidden_size] * num_layers,
            activation_class=nn.Tanh,
            device=device,
        ),
        in_keys=["observation"],
    )


def _parse_device_list(spec):
    return [item.strip() for item in spec.split(",") if item.strip()]


def _resolve_collector_mode(mode):
    if mode == "multiasync":
        return "semi-async"
    return mode


def _linear_schedule(start, end, step, duration):
    if end is None or duration is None or duration <= 0:
        return start
    frac = min(max(step / duration, 0.0), 1.0)
    return start + frac * (end - start)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def train(args):
    env_device = args.device
    train_device = args.train_device or env_device
    collector_mode = _resolve_collector_mode(args.collector_mode)

    # --- Envs ---
    compile_kwargs = {}
    if args.compile_mode:
        compile_kwargs["mode"] = args.compile_mode

    # Bootstrap ObservationNorm eagerly, then rebuild the real training env
    # with compilation enabled for steady-state collection/training.
    init_env = make_env(
        args.env,
        args.num_envs,
        env_device,
        args.frame_skip,
        compile_step=False,
    )
    obs_norm_td = init_obs_norm_stats(init_env, 50)
    init_env.close()

    train_env = make_env(
        args.env,
        args.num_envs,
        env_device,
        args.frame_skip,
        compile_step=args.compile,
        compile_kwargs=compile_kwargs or None,
        obs_norm_td=obs_norm_td,
    )
    obs_dim = train_env.observation_spec["observation"].shape[-1]
    act_dim = train_env.action_spec.shape[-1]

    # --- Logger ---
    logger = WandbLogger(
        exp_name=f"{args.env}_ppo",
        project=args.wandb_project,
        config=vars(args),
    )
    logger.experiment.define_metric("global_step")
    for metric in ("reward/*", "loss/*", "perf/*", "eval/*", "inference/*", "training/*"):
        logger.experiment.define_metric(metric, step_metric="global_step")
    # Share obs normalization stats from the eager bootstrap to eval env.
    mjt_logger.info(f"  Train ObservationNorm state_dict keys={list(obs_norm_td.keys())}")
    for k, v in obs_norm_td.items():
        mjt_logger.info(f"    {k}: shape={v.shape} dtype={v.dtype} device={v.device}")
    # --- Models (on train device) ---
    actor = make_actor(
        obs_dim,
        act_dim,
        train_device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    critic = make_critic(
        obs_dim,
        train_device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    eval_device = args.eval_device or train_device
    eval_env = make_eval_env(
        args.env,
        eval_device,
        args.frame_skip,
        logger,
        obs_norm_td=obs_norm_td,
        compile_step=args.compile,
        compile_kwargs=compile_kwargs or None,
    )
    eval_actor = make_actor(
        obs_dim,
        act_dim,
        eval_device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    evaluator = Evaluator(
        eval_env,
        eval_actor,
        max_steps=1000,
        logger=logger,
    )

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
        normalize_advantage=True,
    )

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=args.lr,
        eps=1e-5,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=train_device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.mini_batch_size,
    )

    # --- Collector ---
    # Env runs on env_device; collected data is stored on train_device.
    async_replay_buffer = None
    if collector_mode == "sync":
        collector = SyncDataCollector(
            train_env,
            actor,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            device=env_device,
            storing_device=train_device,
        )
    elif collector_mode == "semi-async":
        if not args.collector_devices:
            raise ValueError("--collector_devices is required for --collector_mode semi-async")
        collector_devices = _parse_device_list(args.collector_devices)
        num_workers = len(collector_devices)
        if args.num_envs % num_workers:
            raise ValueError("num_envs must be divisible by the number of collector devices")
        if args.frames_per_batch % num_workers:
            raise ValueError("frames_per_batch must be divisible by the number of collector devices")

        num_envs_per_worker = args.num_envs // num_workers
        frames_per_batch_per_worker = args.frames_per_batch // num_workers
        env_fns = [
            partial(
                make_env,
                args.env,
                num_envs_per_worker,
                device,
                args.frame_skip,
                compile_step=args.compile,
                compile_kwargs=compile_kwargs or None,
                obs_norm_td=obs_norm_td,
            )
            for device in collector_devices
        ]
        collector = MultiaSyncDataCollector(
            env_fns,
            actor,
            frames_per_batch=[frames_per_batch_per_worker] * num_workers,
            total_frames=args.total_frames,
            storing_device=train_device,
            env_device=collector_devices,
            policy_device=collector_devices,
            update_at_each_batch=False,
            cat_results="stack",
        )
    elif collector_mode == "full-async":
        collector_device = args.collector_devices
        if collector_device is None:
            collector_device = env_device
        elif "," in collector_device:
            collector_device = _parse_device_list(collector_device)[0]

        async_replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                max_size=max(args.frames_per_batch * 2, args.num_envs * 10),
                ndim=2,
                shared_init=True,
            ),
            sampler=SamplerWithoutReplacement(),
            shared=True,
        )
        collector = aSyncDataCollector(
            partial(
                make_env,
                args.env,
                args.num_envs,
                collector_device,
                args.frame_skip,
                compile_step=args.compile,
                compile_kwargs=compile_kwargs or None,
                obs_norm_td=obs_norm_td,
            ),
            actor,
            replay_buffer=async_replay_buffer,
            frames_per_batch=args.frames_per_batch,
            total_frames=args.total_frames,
            storing_device="cpu",
            env_device=collector_device,
            policy_device=collector_device,
            update_at_each_batch=False,
            local_init_rb=True,
        )
    else:
        raise ValueError(f"Unknown collector_mode: {args.collector_mode}")

    mjt_logger.info(
        f"PPO [{args.env}] obs={obs_dim} act={act_dim}\n"
        f"  env_device={env_device} train_device={train_device}\n"
        f"  num_envs={args.num_envs} frames_per_batch={args.frames_per_batch} "
        f"total_frames={args.total_frames}\n"
        f"  collector_mode={collector_mode}"
        f"{f' collector_devices={args.collector_devices}' if args.collector_devices else ''}\n"
        f"  gamma={args.gamma} gae_lambda={args.gae_lambda} "
        f"clip={args.clip_epsilon} epochs={args.num_epochs}"
    )

    t0 = time.perf_counter()
    best_reward = float("-inf")
    collected_frames = 0
    prev_iter_end = t0

    if collector_mode == "full-async":
        collector.start()
        last_write_count = 0
        iteration = -1
        while collected_frames < args.total_frames:
            iteration += 1
            wait_start = time.perf_counter()
            while True:
                write_count = int(collector.getattr_rb("write_count"))
                if write_count - last_write_count >= args.frames_per_batch:
                    break
                if write_count >= args.total_frames and write_count > last_write_count:
                    break
                time.sleep(0.01)

            iter_start = time.perf_counter()
            collection_time = iter_start - wait_start
            batch_frames = max(1, min(write_count - last_write_count, args.frames_per_batch))
            last_write_count = write_count
            collected_frames = min(write_count, args.total_frames)
            batch = async_replay_buffer.sample(batch_size=args.frames_per_batch).to(train_device)

            current_lr = _linear_schedule(
                args.lr,
                args.lr_final,
                collected_frames,
                args.lr_anneal_frames,
            )
            for group in optim.param_groups:
                group["lr"] = current_lr

            current_entropy_coeff = _linear_schedule(
                args.entropy_coeff,
                args.entropy_coeff_final,
                collected_frames,
                args.entropy_anneal_frames,
            )
            loss_module.entropy_coeff.copy_(
                torch.as_tensor(
                    current_entropy_coeff,
                    device=loss_module.entropy_coeff.device,
                    dtype=loss_module.entropy_coeff.dtype,
                )
            )

            train_start = time.perf_counter()
            sgd_steps = 0
            for _ in range(args.num_epochs):
                with torch.no_grad():
                    adv_module(batch)
                replay_buffer.empty()
                replay_buffer.extend(batch.reshape(-1))
                for mb in replay_buffer:
                    loss_vals = loss_module(mb)
                    total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    optim.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        loss_module.parameters(),
                        args.max_grad_norm,
                    )
                    optim.step()
                    sgd_steps += 1
            train_time = time.perf_counter() - train_start
            collector.update_policy_weights_()

            ep_done = batch["next", "done"].squeeze(-1)
            ep_rews = batch["next", "episode_reward"][ep_done]
            mean_ep = ep_rews.mean().item() if ep_rews.numel() > 0 else float("nan")
            if mean_ep == mean_ep:
                best_reward = max(best_reward, mean_ep)

            eval_time = 0.0
            should_eval = (iteration + 1) % args.eval_interval == 0 or iteration == 0
            if should_eval:
                if args.async_eval:
                    eval_start = time.perf_counter()
                    evaluator.trigger_eval(actor, step=collected_frames)
                    eval_time = time.perf_counter() - eval_start
                    mjt_logger.info(f"    [EVAL] queued on {eval_device}")
                else:
                    eval_start = time.perf_counter()
                    eval_metrics = evaluator.evaluate(actor, step=collected_frames)
                    eval_time = time.perf_counter() - eval_start
                    mjt_logger.info(f"    [EVAL] ep_reward={eval_metrics['eval/reward']:.1f}")

            if args.async_eval:
                eval_result = evaluator.poll()
                if eval_result is not None:
                    mjt_logger.info(f"    [EVAL DONE] step={eval_result['eval/step']} ep_reward={eval_result['eval/reward']:.1f}")

            iter_end = time.perf_counter()
            iter_time = iter_end - iter_start
            total_elapsed = iter_end - t0

            logger.experiment.log(
                {
                    "reward/episode_reward": mean_ep,
                    "reward/best_episode_reward": best_reward,
                    "reward/mean_step_reward": batch["next", "reward"].mean().item(),
                    "loss/objective": loss_vals["loss_objective"].item(),
                    "loss/critic": loss_vals["loss_critic"].item(),
                    "loss/entropy": loss_vals["loss_entropy"].item(),
                    "perf/fps": collected_frames / total_elapsed,
                    "perf/iter_time_s": iter_time,
                    "perf/collection_frac": collection_time / iter_time if iter_time > 0 else 0.0,
                    "perf/training_frac": train_time / iter_time if iter_time > 0 else 0.0,
                    "perf/eval_frac": eval_time / iter_time if iter_time > 0 else 0.0,
                    "inference/time_s": collection_time,
                    "inference/fps": batch_frames / collection_time if collection_time > 0 else 0.0,
                    "training/time_s": train_time,
                    "training/sgd_steps": sgd_steps,
                    "training/sgd_steps_per_s": sgd_steps / train_time if train_time > 0 else 0.0,
                    "training/frames_per_s_equiv": batch_frames / train_time if train_time > 0 else 0.0,
                    "training/lr": current_lr,
                    "training/entropy_coeff": current_entropy_coeff,
                    "eval/time_s": eval_time,
                    "global_step": collected_frames,
                }
            )

            if (iteration + 1) % args.log_interval == 0 or iteration == 0:
                fps = collected_frames / total_elapsed
                mjt_logger.info(
                    f"  iter {iteration + 1} | frames={collected_frames} | "
                    f"ep_reward={mean_ep:.1f} | best={best_reward:.1f} | "
                    f"fps={fps:.0f} | lr={current_lr:.2e} | ent={current_entropy_coeff:.3g} | "
                    f"collect={collection_time:.2f}s | train={train_time:.2f}s"
                )
    else:
        for iteration, batch in enumerate(collector):
            iter_start = time.perf_counter()
            collection_time = iter_start - prev_iter_end
            batch_frames = batch.numel()
            collected_frames += batch_frames

            current_lr = _linear_schedule(
                args.lr,
                args.lr_final,
                collected_frames,
                args.lr_anneal_frames,
            )
            for group in optim.param_groups:
                group["lr"] = current_lr

            current_entropy_coeff = _linear_schedule(
                args.entropy_coeff,
                args.entropy_coeff_final,
                collected_frames,
                args.entropy_anneal_frames,
            )
            loss_module.entropy_coeff.copy_(
                torch.as_tensor(
                    current_entropy_coeff,
                    device=loss_module.entropy_coeff.device,
                    dtype=loss_module.entropy_coeff.dtype,
                )
            )

            train_start = time.perf_counter()
            sgd_steps = 0
            for _ in range(args.num_epochs):
                with torch.no_grad():
                    adv_module(batch)
                replay_buffer.empty()
                replay_buffer.extend(batch.reshape(-1))
                for mb in replay_buffer:
                    loss_vals = loss_module(mb)
                    total_loss = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]
                    optim.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        loss_module.parameters(),
                        args.max_grad_norm,
                    )
                    optim.step()
                    sgd_steps += 1
            train_time = time.perf_counter() - train_start
            collector.update_policy_weights_()

            ep_done = batch["next", "done"].squeeze(-1)
            ep_rews = batch["next", "episode_reward"][ep_done]
            mean_ep = ep_rews.mean().item() if ep_rews.numel() > 0 else float("nan")
            if mean_ep == mean_ep:  # not nan
                best_reward = max(best_reward, mean_ep)

            eval_time = 0.0
            should_eval = (iteration + 1) % args.eval_interval == 0 or iteration == 0
            if should_eval:
                if args.async_eval:
                    eval_start = time.perf_counter()
                    evaluator.trigger_eval(actor, step=collected_frames)
                    eval_time = time.perf_counter() - eval_start
                    mjt_logger.info(f"    [EVAL] queued on {eval_device}")
                else:
                    eval_start = time.perf_counter()
                    eval_metrics = evaluator.evaluate(actor, step=collected_frames)
                    eval_time = time.perf_counter() - eval_start
                    mjt_logger.info(f"    [EVAL] ep_reward={eval_metrics['eval/reward']:.1f}")

            if args.async_eval:
                eval_result = evaluator.poll()
                if eval_result is not None:
                    mjt_logger.info(f"    [EVAL DONE] step={eval_result['eval/step']} ep_reward={eval_result['eval/reward']:.1f}")

            iter_end = time.perf_counter()
            iter_time = iter_end - iter_start
            total_elapsed = iter_end - t0
            prev_iter_end = iter_end

            logger.experiment.log(
                {
                    "reward/episode_reward": mean_ep,
                    "reward/best_episode_reward": best_reward,
                    "reward/mean_step_reward": batch["next", "reward"].mean().item(),
                    "loss/objective": loss_vals["loss_objective"].item(),
                    "loss/critic": loss_vals["loss_critic"].item(),
                    "loss/entropy": loss_vals["loss_entropy"].item(),
                    "perf/fps": collected_frames / total_elapsed,
                    "perf/iter_time_s": iter_time,
                    "perf/collection_frac": collection_time / iter_time if iter_time > 0 else 0.0,
                    "perf/training_frac": train_time / iter_time if iter_time > 0 else 0.0,
                    "perf/eval_frac": eval_time / iter_time if iter_time > 0 else 0.0,
                    "inference/time_s": collection_time,
                    "inference/fps": batch_frames / collection_time if collection_time > 0 else 0.0,
                    "training/time_s": train_time,
                    "training/sgd_steps": sgd_steps,
                    "training/sgd_steps_per_s": sgd_steps / train_time if train_time > 0 else 0.0,
                    "training/frames_per_s_equiv": batch_frames / train_time if train_time > 0 else 0.0,
                    "training/lr": current_lr,
                    "training/entropy_coeff": current_entropy_coeff,
                    "eval/time_s": eval_time,
                    "global_step": collected_frames,
                }
            )

            if (iteration + 1) % args.log_interval == 0 or iteration == 0:
                fps = collected_frames / total_elapsed
                mjt_logger.info(
                    f"  iter {iteration + 1} | frames={collected_frames} | "
                    f"ep_reward={mean_ep:.1f} | best={best_reward:.1f} | "
                    f"fps={fps:.0f} | lr={current_lr:.2e} | ent={current_entropy_coeff:.3g} | "
                    f"collect={collection_time:.2f}s | train={train_time:.2f}s"
                )

    elapsed = time.perf_counter() - t0
    if args.async_eval:
        final_result = evaluator.wait()
        if final_result is not None:
            mjt_logger.info(f"    [EVAL DONE] step={final_result['eval/step']} ep_reward={final_result['eval/reward']:.1f}")
    evaluator.shutdown()
    train_env.close()
    collector.shutdown()
    mjt_logger.info(
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
    p.add_argument("--lr_final", type=float, default=None)
    p.add_argument("--lr_anneal_frames", type=int, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_epsilon", type=float, default=0.2)
    p.add_argument("--entropy_coeff", type=float, default=0.01)
    p.add_argument("--entropy_coeff_final", type=float, default=None)
    p.add_argument("--entropy_anneal_frames", type=int, default=None)
    p.add_argument("--critic_coeff", type=float, default=0.5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--mini_batch_size", type=int, default=4096)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)

    # Logging
    p.add_argument("--log_interval", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--wandb_project", type=str, default="mujoco-torch-zoo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="Env/collection device (default: cuda)")
    p.add_argument("--train_device", type=str, default=None, help="Training device (default: same as --device)")
    p.add_argument("--eval_device", type=str, default=None, help="Evaluation device (default: same as --train_device)")
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument(
        "--collector_mode",
        choices=["sync", "semi-async", "full-async", "multiasync"],
        default="sync",
    )
    p.add_argument("--collector_devices", type=str, default=None)
    p.add_argument("--async_eval", action="store_true", default=False)
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
