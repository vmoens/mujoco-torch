#!/usr/bin/env python3
"""Benchmark throughput at different levels to find the bottleneck.

Compares:
1. Raw vmap(step) — the benchmark number
2. compile(vmap(step)) — what compile_step should give
3. env._step() — full env step with reward, obs, reset
4. Collector iteration — full TorchRL collection loop
"""

import argparse
import time

import torch

import mujoco_torch
from mujoco_torch.zoo import ENVS


def bench_raw_physics(env, num_steps=200):
    """Raw vmap(step) — matches the README benchmark."""
    env.reset()
    mx = env.mx
    dx = env._dx
    step_fn = lambda d: mujoco_torch.step(mx, d)
    vmap_step = torch.vmap(step_fn)

    # Warmup
    for _ in range(5):
        dx = vmap_step(dx)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        dx = vmap_step(dx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = num_steps * env.num_envs
    fps = total_steps / elapsed
    print(f"  Raw vmap(step):           {fps:>12,.0f} steps/sec  ({elapsed:.3f}s for {total_steps:,} steps)")
    return fps


def bench_compiled_physics(env, num_steps=200):
    """compile(vmap(step)) — what compile_step should give."""
    env.reset()
    mx = env.mx
    dx = env._dx
    step_fn = lambda d: mujoco_torch.step(mx, d)
    compiled_vmap_step = torch.compile(torch.vmap(step_fn))

    # Warmup (compile happens here)
    print("  Compiling vmap(step)...", end="", flush=True)
    for _ in range(5):
        dx = compiled_vmap_step(dx)
    torch.cuda.synchronize()
    print(" done")

    t0 = time.perf_counter()
    for _ in range(num_steps):
        dx = compiled_vmap_step(dx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = num_steps * env.num_envs
    fps = total_steps / elapsed
    print(f"  compile(vmap(step)):      {fps:>12,.0f} steps/sec  ({elapsed:.3f}s for {total_steps:,} steps)")
    return fps


def bench_env_step(env, num_steps=200):
    """Full env._step() with reward, obs, reset logic."""
    from tensordict import TensorDict

    # Reset to get initial state
    td = env.reset()
    # Create a dummy action
    action = torch.zeros(env.num_envs, env.action_spec.shape[-1], dtype=env.dtype, device=env.device)

    # Warmup
    for _ in range(5):
        td_in = TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        td_out = env._step(td_in)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        td_in = TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        td_out = env._step(td_in)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_steps = num_steps * env.num_envs
    fps = total_steps / elapsed
    print(f"  env._step():              {fps:>12,.0f} steps/sec  ({elapsed:.3f}s for {total_steps:,} steps)")
    return fps


def bench_env_step_with_frameskip(env, num_steps=200):
    """env._step() accounts for frame_skip — report physics steps/sec."""
    from tensordict import TensorDict

    td = env.reset()
    action = torch.zeros(env.num_envs, env.action_spec.shape[-1], dtype=env.dtype, device=env.device)

    # Warmup
    for _ in range(5):
        td_in = TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        td_out = env._step(td_in)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_steps):
        td_in = TensorDict({"action": action}, batch_size=env.batch_size, device=env.device)
        td_out = env._step(td_in)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    physics_steps = num_steps * env.num_envs * env.FRAME_SKIP
    fps = physics_steps / elapsed
    print(f"  env._step() physics:      {fps:>12,.0f} phys_steps/sec  (frame_skip={env.FRAME_SKIP})")
    return fps


def _make_actor(env, device):
    """Build a small stochastic actor for benchmarking."""
    import torch.nn as nn
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MLP, NormalParamExtractor, ProbabilisticActor, TanhNormal

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim, out_features=2 * act_dim, num_cells=[256, 256], activation_class=nn.ReLU, device=device
        ),
        NormalParamExtractor(),
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    return ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )


def _make_env(env_name, num_envs, device, frame_skip, compile_step, fast=False):
    """Build a TransformedEnv, optionally with fast-path flags."""
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat, RewardSum, StepCounter

    cls = ENVS[env_name]
    base = cls(num_envs=num_envs, device=device, frame_skip=frame_skip, compile_step=compile_step)
    env = TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            StepCounter(max_steps=1000),
            RewardSum(),
        ),
    )
    if fast:
        # Enable fast-path flags from pytorch/rl faster-collector branch
        base._trust_step_output = True
        env._trust_step_output = True
        # NOTE: _skip_maybe_reset requires the env to auto-reset inside _step().
        # MujocoTorchEnv handles resets in _reset() via the collector, so we
        # cannot skip maybe_reset.
    return env


def _bench_collector_inner(env, actor, num_envs, num_frames, label, fast=False):
    """Run a single collector benchmark and print results."""
    try:
        from torchrl.collectors import Collector
    except ImportError:
        from torchrl.collectors import SyncDataCollector as Collector

    collector_kwargs = dict(
        frames_per_batch=num_envs,
        total_frames=num_frames + num_envs * 10,  # extra for warmup
        device=env.device,
    )
    if fast:
        collector_kwargs["update_traj_ids"] = False

    collector = Collector(env, actor, **collector_kwargs)

    # Warmup
    it = iter(collector)
    for _ in range(5):
        batch = next(it)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    count = 0
    for batch in collector:
        count += batch.numel()
        if count >= num_frames:
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = count / elapsed if elapsed > 0 else 0
    print(f"  {label:<30s} {fps:>12,.0f} frames/sec  ({elapsed:.3f}s for {count:,} frames)")

    del collector
    return fps


def bench_collector(env_name, num_envs, device, frame_skip, compile_step, num_frames=500_000):
    """Full TorchRL collector loop (standard flags)."""
    env = _make_env(env_name, num_envs, device, frame_skip, compile_step, fast=False)
    actor = _make_actor(env, device)
    fps = _bench_collector_inner(env, actor, num_envs, num_frames, "Collector (standard):", fast=False)
    del env
    return fps


def bench_collector_fast(env_name, num_envs, device, frame_skip, compile_step, num_frames=500_000):
    """TorchRL collector with faster-collector branch optimizations."""
    env = _make_env(env_name, num_envs, device, frame_skip, compile_step, fast=True)
    actor = _make_actor(env, device)
    fps = _bench_collector_inner(env, actor, num_envs, num_frames, "Collector (fast flags):", fast=True)
    del env
    return fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="humanoid_rich")
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_steps", type=int, default=200)
    p.add_argument(
        "--collector_only", action="store_true", help="Skip raw physics tests, only run collector benchmarks"
    )
    args = p.parse_args()

    print(
        f"\n=== Throughput benchmark: {args.env}, {args.num_envs} envs, "
        f"frame_skip={args.frame_skip}, device={args.device} ===\n"
    )

    if not args.collector_only:
        # Create base env (no compile)
        cls = ENVS[args.env]
        env_eager = cls(num_envs=args.num_envs, device=args.device, frame_skip=args.frame_skip, compile_step=False)

        print("[1] Raw physics (no compile)")
        bench_raw_physics(env_eager, args.num_steps)
        bench_env_step(env_eager, args.num_steps)
        bench_env_step_with_frameskip(env_eager, args.num_steps)

        del env_eager

        print("\n[2] Compiled physics")
        env_compiled = cls(num_envs=args.num_envs, device=args.device, frame_skip=args.frame_skip, compile_step=True)
        bench_compiled_physics(env_compiled, args.num_steps)
        bench_env_step(env_compiled, args.num_steps)
        bench_env_step_with_frameskip(env_compiled, args.num_steps)

        del env_compiled

    nf = args.num_envs * 200

    print("\n[3] Collector (compiled, standard)")
    bench_collector(args.env, args.num_envs, args.device, args.frame_skip, compile_step=True, num_frames=nf)

    print("\n[4] Collector (compiled, fast flags)")
    bench_collector_fast(args.env, args.num_envs, args.device, args.frame_skip, compile_step=True, num_frames=nf)

    print("\n[5] Collector (eager, fast flags)")
    bench_collector_fast(args.env, args.num_envs, args.device, args.frame_skip, compile_step=False, num_frames=nf)

    print("\nDone.")


if __name__ == "__main__":
    main()
