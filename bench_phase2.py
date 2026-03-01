"""Phase 2 benchmark: test code-level optimizations (H3+H5, H6, H8).

Run on cluster:
  python -u bench_phase2.py
"""

import gc
import shutil
import time

import mujoco
import numpy as np
import torch
import torch._inductor.config as inductor_config

import mujoco_torch
from mujoco_torch._src.test_util import load_test_file


def clear_compile_caches():
    """Wipe dynamo + inductor caches so the next compile starts fresh."""
    torch._dynamo.reset()
    torch.compiler.reset()
    gc.collect()
    torch.cuda.empty_cache()
    for candidate in [
        getattr(torch._inductor.config, "cache_dir", None),
        "/tmp/torchinductor_root",
    ]:
        if candidate:
            shutil.rmtree(candidate, ignore_errors=True)

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
    clear_compile_caches()

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

    return sps


def main():
    m_mj = load_test_file("humanoid.xml")
    torch.set_default_dtype(torch.float64)

    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: humanoid, B={B}, NSTEPS={NSTEPS}", flush=True)

    results = {}

    mx = mujoco_torch.device_put(m_mj).to(DEVICE)
    mx_f32 = mujoco_torch.device_put(m_mj, dtype=torch.float32).to(DEVICE)

    # (key, label, model, compile_kwargs, step_kwargs, dtype, inductor_settings)
    hypotheses = [
        ("baseline", "Baseline (H3+H5 active)", mx,
         dict(fullgraph=True), {}, torch.float64, {}),
        ("h4_inductor", "H4: inductor config", mx,
         dict(fullgraph=True), {}, torch.float64,
         {"coordinate_descent_tuning": True, "aggressive_fusion": True}),
        ("h8_fixed", "H8: fixed_iterations", mx,
         dict(fullgraph=True), dict(fixed_iterations=True), torch.float64, {}),
        ("h8_reduce", "H8 + reduce-overhead", mx,
         dict(fullgraph=True, mode="reduce-overhead"),
         dict(fixed_iterations=True), torch.float64, {}),
        ("h4_h8_reduce", "H4+H8 + reduce-overhead", mx,
         dict(fullgraph=True, mode="reduce-overhead"),
         dict(fixed_iterations=True), torch.float64,
         {"coordinate_descent_tuning": True, "aggressive_fusion": True}),
        ("h6_f32", "H6: float32", mx_f32,
         dict(fullgraph=True), {}, torch.float32, {}),
        ("all_f64", "ALL f64: H4+H8+reduce-overhead", mx,
         dict(fullgraph=True, mode="reduce-overhead"),
         dict(fixed_iterations=True), torch.float64,
         {"coordinate_descent_tuning": True, "aggressive_fusion": True}),
        ("all_f32", "ALL f32: H4+H6+H8+reduce-overhead", mx_f32,
         dict(fullgraph=True, mode="reduce-overhead"),
         dict(fixed_iterations=True), torch.float32,
         {"coordinate_descent_tuning": True, "aggressive_fusion": True}),
    ]

    for key, label, model, compile_kwargs, step_kwargs, dtype, ind_cfg in hypotheses:
        torch.set_default_dtype(dtype)
        for k, v in ind_cfg.items():
            setattr(inductor_config, k, v)
        try:
            sps = run_benchmark(label, model, m_mj, compile_kwargs,
                                step_kwargs=step_kwargs, dtype=dtype)
            results[key] = sps
        except Exception as e:
            print(f"  {label}: FAILED — {e}", flush=True)
            results[key] = None
        torch.set_default_dtype(torch.float64)
        for k in ind_cfg:
            setattr(inductor_config, k, False)

    # Summary
    print("\n" + "=" * 70)
    print("  PHASE 2 RESULTS SUMMARY")
    print("=" * 70)
    baseline = results.get("baseline")
    for k, v in results.items():
        if v is None:
            print(f"  {k:50s}  {'FAILED':>12s}")
        elif baseline:
            delta = (v / baseline - 1) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  {k:50s}  {v:>12,.0f} steps/s  ({sign}{delta:.1f}%)")
        else:
            print(f"  {k:50s}  {v:>12,.0f} steps/s")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
