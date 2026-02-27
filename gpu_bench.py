"""GPU benchmark: mujoco-torch vs MuJoCo C vs MJX (JAX).

Benchmarks a single model at various batch sizes. Configs that scale linearly
(MuJoCo C loop, mujoco-torch sequential loop) are only benchmarked at B=1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model ant
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model humanoid --batch-sizes 1 128 1024 4096
"""

import argparse
import gc
import json
import time
import traceback

import mujoco
import numpy as np
import torch
from etils import epath

import mujoco_torch

torch.set_default_dtype(torch.float64)

DEVICE = "cuda"
WARMUP_ITERS = 5
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="mujoco-torch GPU benchmark")
    parser.add_argument("--model", default="humanoid", help="Model to benchmark (default: humanoid)")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 128, 1024, 4096], help="Batch sizes")
    parser.add_argument("--nsteps", type=int, default=1000, help="Steps per timing run")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file (default: bench_<model>.json)")
    parser.add_argument(
        "--only", type=str, default=None, help="Run only a specific benchmark (compile, c, loop, vmap, mjx)"
    )
    return parser.parse_args()


def load_model_xml(name):
    return (epath.resource_path("mujoco_torch") / "test_data" / f"{name}.xml").read_text()


def make_batched_qvel(nv, batch_size, seed):
    rng = np.random.RandomState(seed)
    return 0.01 * rng.randn(batch_size, nv)


def make_batch(mx, m_mj, B, seed=SEED):
    qvels = make_batched_qvel(m_mj.nv, B, seed)
    envs = []
    with torch.device("cpu"):
        for i in range(B):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d).to(DEVICE))
    return torch.stack(envs, dim=0)


def bench_mujoco_c(m_mj, nsteps):
    """MuJoCo C sequential at B=1 (scales linearly)."""
    d = mujoco.MjData(m_mj)
    d.qvel[:] = make_batched_qvel(m_mj.nv, 1, SEED)[0]

    for _ in range(10):
        mujoco.mj_step(m_mj, d)

    d = mujoco.MjData(m_mj)
    d.qvel[:] = make_batched_qvel(m_mj.nv, 1, SEED)[0]
    t0 = time.perf_counter()
    for _ in range(nsteps):
        mujoco.mj_step(m_mj, d)
    elapsed = time.perf_counter() - t0
    sps = nsteps / elapsed
    return {"B=1": {"elapsed_s": elapsed, "steps_per_s": sps}}


def bench_torch_loop(mx, m_mj, nsteps):
    """mujoco-torch sequential loop at B=1 (scales linearly)."""
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = make_batched_qvel(m_mj.nv, 1, SEED)[0]
    dx = mujoco_torch.device_put(d_mj).to(DEVICE)

    for _ in range(5):
        dx = mujoco_torch.step(mx, dx)
    torch.cuda.synchronize()

    dx = mujoco_torch.device_put(d_mj).to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(nsteps):
        dx = mujoco_torch.step(mx, dx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sps = nsteps / elapsed
    return {"B=1": {"elapsed_s": elapsed, "steps_per_s": sps}}


def bench_vmap(mx, m_mj, batch_sizes, nsteps):
    """mujoco-torch vmap (eager) across batch sizes."""
    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
    results = {}
    for B in batch_sizes:
        d_batch = make_batch(mx, m_mj, B)
        d_batch = vmap_step(d_batch)
        torch.cuda.synchronize()

        d_batch = make_batch(mx, m_mj, B)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nsteps):
            d_batch = vmap_step(d_batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        sps = B * nsteps / elapsed
        results[f"B={B}"] = {"elapsed_s": elapsed, "steps_per_s": sps}
        print(f"    B={B:5d}: {elapsed * 1e3:8.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)
    return results


def bench_compile(mx, m_mj, batch_sizes, nsteps):
    """mujoco-torch compile(fullgraph=True) across batch sizes."""
    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
    # One eager call to populate _CachedConst caches for the target device,
    # so the first torch.compile trace never records a DeviceCopy.
    d_warm = make_batch(mx, m_mj, 1)
    vmap_step(d_warm)
    del d_warm
    results = {}
    for B in batch_sizes:
        torch._dynamo.reset()
        torch.compiler.reset()
        compiled_fn = torch.compile(vmap_step, fullgraph=True)
        d_batch = make_batch(mx, m_mj, B)
        print(f"    B={B:5d}: compiling...", end="", flush=True)
        try:
            for _ in range(WARMUP_ITERS):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()
            print(" timing...", end="", flush=True)

            d_batch = make_batch(mx, m_mj, B)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(nsteps):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            sps = B * nsteps / elapsed
            results[f"B={B}"] = {"elapsed_s": elapsed, "steps_per_s": sps}
            print(f" {elapsed * 1e3:8.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)
        except Exception:
            print(" FAILED", flush=True)
            traceback.print_exc()
            results[f"B={B}"] = None
    return results


def bench_mjx(m_mj, batch_sizes, nsteps):
    """MJX jit(vmap(step)) across batch sizes."""
    try:
        import jax
        from mujoco import mjx
    except ImportError:
        print("  JAX not installed -- skipping MJX comparison.", flush=True)
        return None

    jax.config.update("jax_enable_x64", True)
    mx_jax = mjx.put_model(m_mj)
    step_jax_vmap = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    results = {}
    for B in batch_sizes:
        try:
            d_ref = mujoco.MjData(m_mj)
            dx_single = mjx.put_data(m_mj, d_ref)

            def _tile_leaf(x):
                if not hasattr(x, "ndim"):
                    return x
                if x.ndim == 0:
                    return jax.numpy.broadcast_to(x, (B,))
                return jax.numpy.tile(x, (B,) + (1,) * x.ndim)

            dx_jax = jax.tree.map(_tile_leaf, dx_single)
            qvels = make_batched_qvel(m_mj.nv, B, SEED)
            dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))

            dx_warm = step_jax_vmap(mx_jax, dx_jax)
            jax.block_until_ready(dx_warm.qpos)

            dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))
            t0 = time.perf_counter()
            for _ in range(nsteps):
                dx_jax = step_jax_vmap(mx_jax, dx_jax)
            jax.block_until_ready(dx_jax.qpos)
            elapsed = time.perf_counter() - t0
            sps = B * nsteps / elapsed
            results[f"B={B}"] = {"elapsed_s": elapsed, "steps_per_s": sps}
            print(f"    B={B:5d}: {elapsed * 1e3:8.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)
        except Exception:
            print(f"    B={B:5d}: FAILED", flush=True)
            traceback.print_exc()
            results[f"B={B}"] = None
    return results


def print_summary(model_name, all_results, batch_sizes):
    """Print a clean summary table."""
    print()
    print("=" * 100)
    print(f"  Summary: {model_name}")
    print("=" * 100)

    configs = list(all_results.keys())
    header = f"{'Config':40s}"
    for B in batch_sizes:
        header += f"  {'B=' + str(B):>14s}"
    print(header)
    print("-" * len(header))

    for config in configs:
        data = all_results[config]
        if data is None:
            continue
        row = f"{config:40s}"
        for B in batch_sizes:
            key = f"B={B}"
            if key in data and data[key] is not None:
                row += f"  {data[key]['steps_per_s']:>14,.0f}"
            elif "B=1" in data and data["B=1"] is not None:
                row += f"  {'(linear)':>14s}"
            else:
                row += f"  {'--':>14s}"
        print(row)
    print()


def main():
    args = parse_args()
    model_name = args.model
    output = args.output or f"bench_{model_name}.json"

    print()
    print("#" * 100)
    print(f"#  Model: {model_name}")
    print("#" * 100)

    torch.set_default_device(DEVICE)

    xml = load_model_xml(model_name)
    m_mj = mujoco.MjModel.from_xml_string(xml)
    mx = mujoco_torch.device_put(m_mj).to(DEVICE)

    model_results = {}
    only = args.only

    if only in (None, "compile"):
        print("\n  torch compile(fullgraph=True):", flush=True)
        model_results["torch compile"] = bench_compile(mx, m_mj, args.batch_sizes, args.nsteps)

    if only in (None, "c"):
        print("\n  MuJoCo C (sequential, B=1):", flush=True)
        r = bench_mujoco_c(m_mj, args.nsteps)
        sps = r["B=1"]["steps_per_s"]
        print(f"    B=    1: {r['B=1']['elapsed_s'] * 1e3:8.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)
        model_results["MuJoCo C (seq)"] = r

    if only in (None, "loop"):
        print("\n  mujoco-torch loop (B=1):", flush=True)
        r = bench_torch_loop(mx, m_mj, args.nsteps)
        sps = r["B=1"]["steps_per_s"]
        print(f"    B=    1: {r['B=1']['elapsed_s'] * 1e3:8.1f} ms  ({sps:>12,.0f} steps/s)", flush=True)
        model_results["torch loop (seq)"] = r

    if only in (None, "vmap"):
        print("\n  mujoco-torch vmap (eager):", flush=True)
        model_results["torch vmap (eager)"] = bench_vmap(mx, m_mj, args.batch_sizes, args.nsteps)

    if only in (None, "mjx"):
        gc.collect()
        torch.cuda.empty_cache()
        print("\n  MJX jit(vmap(step)):", flush=True)
        mjx_r = bench_mjx(m_mj, args.batch_sizes, args.nsteps)
        if mjx_r is not None:
            model_results["MJX jit(vmap)"] = mjx_r

    print_summary(model_name, model_results, args.batch_sizes)

    with open(output, "w") as f:
        json.dump({model_name: model_results}, f, indent=2)
    print(f"Results written to {output}", flush=True)


if __name__ == "__main__":
    main()
