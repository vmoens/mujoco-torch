#!/usr/bin/env python3
"""Profile the SyncDataCollector inner loop with torch.profiler.

Runs a short collector rollout (100 steps) and saves a Chrome trace.
"""

import argparse
import torch
from torch.profiler import profile, ProfilerActivity, schedule

from mujoco_torch.zoo import ENVS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="humanoid_rich")
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--frame_skip", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="collector_trace.json.gz")
    p.add_argument("--fast", action="store_true", help="Enable fast-path flags")
    args = p.parse_args()

    # -- env ---------------------------------------------------------------
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import Compose, DoubleToFloat, StepCounter, RewardSum

    cls = ENVS[args.env]
    base = cls(
        num_envs=args.num_envs,
        device=args.device,
        frame_skip=args.frame_skip,
        compile_step=False,
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

    # -- actor -------------------------------------------------------------
    from torchrl.modules import MLP, NormalParamExtractor, ProbabilisticActor, TanhNormal
    from tensordict.nn import TensorDictModule
    import torch.nn as nn

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
    actor_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )

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

    # -- warmup (outside profiler) -----------------------------------------
    print("Warming up (5 steps)...", flush=True)
    it = iter(collector)
    for i in range(5):
        _ = next(it)
        print(f"  warmup step {i+1}/5", flush=True)
    torch.cuda.synchronize()
    print("Warmup done.", flush=True)

    # -- profile 100 collection steps --------------------------------------
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

    # Print a compact summary (no stack grouping to avoid OOM)
    print("\n=== Top 30 by CUDA time ===")
    print(
        prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    )
    print("\n=== Top 30 by CPU time ===")
    print(
        prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
