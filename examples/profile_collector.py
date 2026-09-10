#!/usr/bin/env python3
"""Profile the SyncDataCollector inner loop with torch.profiler.

Instruments key methods with record_function markers for readable traces.
"""

import argparse
import functools

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mujoco_torch.zoo import ENVS


def _wrap_method(obj, method_name, label):
    """Monkey-patch a method with a record_function marker."""
    original = getattr(obj, method_name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)

    setattr(obj, method_name, wrapper)


def _instrument_env(base_env, transformed_env):
    """Add profiler markers to env methods."""
    # Base env methods
    _wrap_method(base_env, "_step", "ENV._step")
    _wrap_method(base_env, "_reset", "ENV._reset")
    _wrap_method(base_env, "_build_obs", "ENV._build_obs")
    _wrap_method(base_env, "_compute_reward", "ENV._compute_reward")
    _wrap_method(base_env, "_compute_terminated", "ENV._compute_terminated")

    # Wrap physics step (the hot path)
    orig_physics = base_env._physics_step

    @functools.wraps(orig_physics)
    def physics_wrapper(*a, **kw):
        with record_function("ENV._physics_step"):
            return orig_physics(*a, **kw)

    base_env._physics_step = physics_wrapper

    # TransformedEnv methods
    _wrap_method(transformed_env, "_step", "TransformedEnv._step")
    _wrap_method(transformed_env, "step", "TransformedEnv.step")

    # Wrap individual transforms
    for i, t in enumerate(transformed_env.transform):
        name = type(t).__name__
        _wrap_method(t, "_step", f"Transform[{i}].{name}._step")


def _instrument_collector(collector):
    """Add profiler markers to collector methods."""
    _wrap_method(collector, "rollout", "Collector.rollout")
    # The _step_and_maybe_reset is the core inner loop
    if hasattr(collector, "_step_and_maybe_reset"):
        _wrap_method(collector, "_step_and_maybe_reset", "Collector._step_and_maybe_reset")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="humanoid_rich")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="collector_trace.json.gz")
    p.add_argument("--fast", action="store_true", help="Enable fast-path flags")
    p.add_argument("--compile", action="store_true", help="Enable compiled physics step")
    p.add_argument("--auto_reset", action="store_true", help="Fuse reset into env._step")
    args = p.parse_args()

    # -- env ---------------------------------------------------------------
    from torchrl.envs import EnvBase, TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat, RewardSum, StepCounter

    cls = ENVS[args.env]
    base = cls(
        num_envs=args.num_envs,
        device=args.device,
        frame_skip=args.frame_skip,
        compile_step=args.compile,
        auto_reset=args.auto_reset,
    )
    env = TransformedEnv(
        base,
        Compose(
            DoubleToFloat(in_keys=["observation"], in_keys_inv=[]),
            StepCounter(max_steps=1000),
            RewardSum(),
        ),
    )
    if args.fast:
        base._trust_step_output = True
        env._trust_step_output = True
    # With auto_reset, _reset is a no-op (env already reset in _step).
    # We keep maybe_reset so InitTracker/transforms work correctly.

    # -- Instrument EnvBase.step and step_and_maybe_reset at class level ---
    orig_envbase_step = EnvBase.step

    @functools.wraps(orig_envbase_step)
    def envbase_step_wrapper(self, *a, **kw):
        with record_function("EnvBase.step"):
            return orig_envbase_step(self, *a, **kw)

    EnvBase.step = envbase_step_wrapper

    if hasattr(EnvBase, "step_and_maybe_reset"):
        orig_samr = EnvBase.step_and_maybe_reset

        @functools.wraps(orig_samr)
        def samr_wrapper(self, *a, **kw):
            with record_function("EnvBase.step_and_maybe_reset"):
                return orig_samr(self, *a, **kw)

        EnvBase.step_and_maybe_reset = samr_wrapper

    if hasattr(EnvBase, "maybe_reset"):
        orig_mr = EnvBase.maybe_reset

        @functools.wraps(orig_mr)
        def mr_wrapper(self, *a, **kw):
            with record_function("EnvBase.maybe_reset"):
                return orig_mr(self, *a, **kw)

        EnvBase.maybe_reset = mr_wrapper

    # Also instrument _step_proc_data if it exists
    if hasattr(EnvBase, "_step_proc_data"):
        orig_spd = EnvBase._step_proc_data

        @functools.wraps(orig_spd)
        def spd_wrapper(self, *a, **kw):
            with record_function("EnvBase._step_proc_data"):
                return orig_spd(self, *a, **kw)

        EnvBase._step_proc_data = spd_wrapper

    # -- Instrument instance methods ---------------------------------------
    _instrument_env(base, env)

    # -- actor -------------------------------------------------------------
    import torch.nn as nn
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MLP, NormalParamExtractor, ProbabilisticActor, TanhNormal

    obs_dim = env.observation_spec["observation"].shape[-1]
    act_dim = env.action_spec.shape[-1]

    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * act_dim,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
            device=args.device,
        ),
        NormalParamExtractor(),
    )
    actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    _wrap_method(actor, "forward", "Actor.forward")

    # -- collector ---------------------------------------------------------
    try:
        from torchrl.collectors import Collector
    except ImportError:
        from torchrl.collectors import SyncDataCollector as Collector

    collector_kwargs = dict(
        frames_per_batch=args.num_envs,  # 1 step per batch
        total_frames=args.num_envs * 500,  # enough headroom
        device=args.device,
    )
    if args.fast:
        collector_kwargs["update_traj_ids"] = False

    collector = Collector(env, actor, **collector_kwargs)
    _instrument_collector(collector)

    # -- warmup (outside profiler) -----------------------------------------
    warmup_steps = 20 if args.compile else 5
    print(f"Warming up ({warmup_steps} steps, compile={args.compile})...", flush=True)
    it = iter(collector)
    for i in range(warmup_steps):
        _ = next(it)
        print(f"  warmup step {i + 1}/{warmup_steps}", flush=True)
    torch.cuda.synchronize()
    print("Warmup done.", flush=True)

    # -- profile -----------------------------------------------------------
    num_profile_steps = 50
    print(f"Profiling {num_profile_steps} collector steps ({args.num_envs} envs)...", flush=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(num_profile_steps):
            _ = next(it)
        torch.cuda.synchronize()

    # -- export ------------------------------------------------------------
    print(f"Exporting trace to {args.output} ...", flush=True)
    prof.export_chrome_trace(args.output)
    print("Trace exported.", flush=True)

    # Print a compact summary
    print("\n=== Top 30 by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))

    print("\nDone.")


if __name__ == "__main__":
    main()
