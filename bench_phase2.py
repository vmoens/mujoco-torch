"""Phase 2 benchmark: test code-level optimizations (H3+H5, H6, H8).

Run on cluster:
  python -u bench_phase2.py
"""

import gc
import time

import mujoco
import numpy as np
import torch
import torch._inductor.config as inductor_config

import mujoco_torch
from mujoco_torch._src.test_util import load_test_file

DEVICE = "cuda"
B = 32768
NSTEPS = 100
SEED = 42


def make_batch(mx, m_mj, batch_size, seed, dtype=torch.float64):
    rng = np.random.RandomState(seed)
    envs = []
    with torch.device("cpu"):
        for i in range(batch_size):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = 0.01 * rng.randn(m_mj.nv)
            envs.append(mujoco_torch.device_put(d).to(DEVICE))
    d_batch = torch.stack(envs, dim=0)
    if dtype != torch.float64:
        d_batch = d_batch.to(dtype)
    return d_batch


def run_benchmark(label, mx, m_mj, compile_kwargs, step_kwargs=None,
                  dtype=torch.float64):
    step_kwargs = step_kwargs or {}
    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d, **step_kwargs))

    print(f"\n{'=' * 70}", flush=True)
    print(f"  {label}: compiling (B={B})...", flush=True)

    compiled_fn = torch.compile(vmap_step, **compile_kwargs)
    d_batch = make_batch(mx, m_mj, B, SEED, dtype=dtype)
    compiled_fn(d_batch)
    torch.cuda.synchronize()
    print(f"  {label}: compiled. warming up...", flush=True)

    for _ in range(5):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()

    d_batch = make_batch(mx, m_mj, B, SEED, dtype=dtype)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    sps = B * NSTEPS / wall
    print(f"  {label}: {wall * 1e3:.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()

    return sps


def main():
    m_mj = load_test_file("humanoid.xml")
    torch.set_default_dtype(torch.float64)

    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: humanoid, B={B}, NSTEPS={NSTEPS}", flush=True)

    results = {}

    # 1. Baseline (H3+H5 code already active)
    mx = mujoco_torch.device_put(m_mj).to(DEVICE)
    sps = run_benchmark(
        "Baseline (H3+H5 active)",
        mx, m_mj,
        dict(fullgraph=True),
    )
    results["baseline_h3h5"] = sps

    # 2. H8: fixed_iterations=True
    sps = run_benchmark(
        "H8: fixed_iterations=True",
        mx, m_mj,
        dict(fullgraph=True),
        step_kwargs=dict(fixed_iterations=True),
    )
    results["h8_fixed_iter"] = sps

    # 3. H8 + mode=reduce-overhead (CUDA graph capture should now work)
    sps = run_benchmark(
        "H8 + reduce-overhead",
        mx, m_mj,
        dict(fullgraph=True, mode="reduce-overhead"),
        step_kwargs=dict(fixed_iterations=True),
    )
    results["h8_reduce_overhead"] = sps

    # 4. H6: float32 mode
    mx_f32 = mujoco_torch.device_put(m_mj, dtype=torch.float32).to(DEVICE)
    torch.set_default_dtype(torch.float32)
    sps = run_benchmark(
        "H6: float32",
        mx_f32, m_mj,
        dict(fullgraph=True),
        dtype=torch.float32,
    )
    results["h6_float32"] = sps

    # 5. H6+H8: float32 + fixed_iterations + reduce-overhead
    sps = run_benchmark(
        "H6+H8: float32 + fixed_iter + reduce-overhead",
        mx_f32, m_mj,
        dict(fullgraph=True, mode="reduce-overhead"),
        step_kwargs=dict(fixed_iterations=True),
        dtype=torch.float32,
    )
    results["h6_h8_combined"] = sps

    # 6. Best combined: float32 + fixed_iter + reduce-overhead + inductor tuning
    inductor_config.coordinate_descent_tuning = True
    inductor_config.aggressive_fusion = True
    sps = run_benchmark(
        "ALL: f32 + fixed + reduce-overhead + inductor",
        mx_f32, m_mj,
        dict(fullgraph=True, mode="reduce-overhead"),
        step_kwargs=dict(fixed_iterations=True),
        dtype=torch.float32,
    )
    results["all_combined"] = sps
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aggressive_fusion = False
    torch.set_default_dtype(torch.float64)

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 2 RESULTS SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:50s}  {v:>12,.0f} steps/s")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
