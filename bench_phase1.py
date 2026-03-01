"""Phase 0+1 benchmark: profiling + quick-win hypotheses.

Runs on humanoid B=32768, NSTEPS=100 for quick iteration.
Tests: baseline, reduce-overhead, matmul precision, inductor config.
Also collects a profiler trace.
"""

import gc
import shutil
import time

import mujoco
import numpy as np
import torch
import torch._inductor.config as inductor_config

import mujoco_torch


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

torch.set_default_dtype(torch.float64)

DEVICE = "cuda"
MODEL = "humanoid"
B = 32768
NSTEPS = 100
WARMUP = 5
SEED = 42


def load_model():
    from etils import epath
    xml = (epath.resource_path("mujoco_torch") / "test_data" / f"{MODEL}.xml").read_text()
    return mujoco.MjModel.from_xml_string(xml)


def make_batch(mx, m_mj):
    rng = np.random.RandomState(SEED)
    qvels = 0.01 * rng.randn(B, m_mj.nv)
    envs = []
    with torch.device("cpu"):
        for i in range(B):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d).to(DEVICE))
    return torch.stack(envs, dim=0)


def warm_caches(mx, m_mj):
    d_mj = mujoco.MjData(m_mj)
    dx = mujoco_torch.device_put(d_mj).to(DEVICE)
    mujoco_torch.step(mx, dx)
    torch.cuda.synchronize()


def run_benchmark(label, mx, m_mj, compile_kwargs):
    clear_compile_caches()

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
    compiled_fn = torch.compile(vmap_step, **compile_kwargs)

    d_batch = make_batch(mx, m_mj)
    print(f"\n{'='*70}", flush=True)
    print(f"  {label}: compiling (B={B})...", flush=True)

    for _ in range(WARMUP):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()

    d_batch = make_batch(mx, m_mj)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    sps = B * NSTEPS / elapsed
    print(f"  {label}: {sps:,.0f} steps/s ({elapsed:.2f}s)", flush=True)
    return sps, compiled_fn, d_batch


def run_profiling(compiled_fn, mx, m_mj):
    print(f"\n{'='*70}", flush=True)
    print("  Profiling (10 steps)...", flush=True)

    d_batch = make_batch(mx, m_mj)
    for _ in range(WARMUP):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()

    d_batch = make_batch(mx, m_mj)
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            d_batch = compiled_fn(d_batch)
        torch.cuda.synchronize()

    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    except KeyError:
        print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=50))
    prof.export_chrome_trace("/tmp/trace_humanoid_32k.json")
    print("  Trace saved to /tmp/trace_humanoid_32k.json", flush=True)

    events = prof.key_averages()
    total_kernels = len(events)

    def _cuda_time(e):
        return getattr(e, "cuda_time_total", getattr(e, "device_time_total", 0))

    total_cuda_time = sum(_cuda_time(e) for e in events)
    print(f"\n  Total unique kernel types: {total_kernels}", flush=True)
    print(f"  Total CUDA time (10 steps): {total_cuda_time / 1e3:.1f} ms", flush=True)
    if total_cuda_time > 0:
        print("\n  Top 10 by CUDA time:", flush=True)
        sorted_events = sorted(events, key=_cuda_time, reverse=True)
        for i, e in enumerate(sorted_events[:10]):
            pct = 100 * _cuda_time(e) / total_cuda_time
            print(f"    {i+1}. {e.key[:60]:60s}  {_cuda_time(e)/1e3:8.2f} ms ({pct:5.1f}%)  count={e.count}", flush=True)


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Model: {MODEL}, B={B}, NSTEPS={NSTEPS}", flush=True)

    m_mj = load_model()
    mx = mujoco_torch.device_put(m_mj).to(DEVICE)
    torch.set_default_device(DEVICE)
    warm_caches(mx, m_mj)

    results = {}

    # Baseline
    sps, compiled_fn, _ = run_benchmark(
        "Baseline (fullgraph=True)",
        mx, m_mj,
        dict(fullgraph=True),
    )
    results["baseline"] = sps

    # H9: Profile — skipped for now (triggers recompilation under profiler context)
    # run_profiling(compiled_fn, mx, m_mj)

    # H1: reduce-overhead
    sps, _, _ = run_benchmark(
        "H1: reduce-overhead",
        mx, m_mj,
        dict(fullgraph=True, mode="reduce-overhead"),
    )
    results["h1_reduce_overhead"] = sps

    # H2: matmul precision
    torch.set_float32_matmul_precision("high")
    sps, _, _ = run_benchmark(
        "H2: matmul precision (high)",
        mx, m_mj,
        dict(fullgraph=True),
    )
    results["h2_matmul_precision"] = sps
    torch.set_float32_matmul_precision("highest")

    # H4: inductor config
    inductor_config.coordinate_descent_tuning = True
    inductor_config.aggressive_fusion = True
    sps, _, _ = run_benchmark(
        "H4: inductor config (cd_tuning + aggressive_fusion)",
        mx, m_mj,
        dict(fullgraph=True),
    )
    results["h4_inductor_config"] = sps
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aggressive_fusion = False

    # H1+H4 combined
    inductor_config.coordinate_descent_tuning = True
    inductor_config.aggressive_fusion = True
    sps, _, _ = run_benchmark(
        "H1+H4: reduce-overhead + inductor config",
        mx, m_mj,
        dict(fullgraph=True, mode="reduce-overhead"),
    )
    results["h1_h4_combined"] = sps
    inductor_config.coordinate_descent_tuning = False
    inductor_config.aggressive_fusion = False

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    baseline = results["baseline"]
    for name, sps in results.items():
        delta = (sps / baseline - 1) * 100
        sign = "+" if delta >= 0 else ""
        print(f"  {name:45s}  {sps:>12,.0f} steps/s  ({sign}{delta:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
