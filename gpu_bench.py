"""GPU benchmark: mujoco-torch vs MuJoCo C vs MJX (JAX).

Benchmarks one or more models at various batch sizes. Configs that scale
linearly (MuJoCo C loop, mujoco-torch sequential loop) are only benchmarked
at B=1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model ant
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model all
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model humanoid halfcheetah
    CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py --model humanoid --batch-sizes 1 128 1024 4096
"""

import argparse
import gc
import json
import logging
import time
import traceback

import mujoco
import numpy as np
import torch
from etils import epath

import mujoco_torch

torch.set_default_dtype(torch.float64)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEVICE = "cuda"
WARMUP_ITERS = 5
SEED = 42
ALL_MODELS = ["humanoid", "ant", "halfcheetah", "walker2d", "hopper"]


def parse_args():
    parser = argparse.ArgumentParser(description="mujoco-torch GPU benchmark")
    parser.add_argument(
        "--model",
        nargs="+",
        default=["humanoid"],
        help=f"Model(s) to benchmark, or 'all' for {ALL_MODELS} (default: humanoid)",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32768, 4096, 1024, 128, 1], help="Batch sizes")
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
        log.info("    B=%5d: %8.1f ms  (%14s steps/s)", B, elapsed * 1e3, f"{sps:,.0f}")
    return results


def _warm_caches(mx, m_mj):
    """Single non-vmapped step to populate _CachedConst caches on the target device."""
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = make_batched_qvel(m_mj.nv, 1, SEED)[0]
    dx = mujoco_torch.device_put(d_mj).to(DEVICE)
    mujoco_torch.step(mx, dx)
    torch.cuda.synchronize()


def bench_compile(mx, m_mj, batch_sizes, nsteps):
    """mujoco-torch compile(fullgraph=True) across batch sizes."""
    _warm_caches(mx, m_mj)
    results = {}
    for B in batch_sizes:
        torch._dynamo.reset()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
        vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
        compiled_fn = torch.compile(vmap_step, fullgraph=True)
        d_batch = make_batch(mx, m_mj, B)
        log.info("    B=%5d: compiling...", B)
        try:
            for _ in range(WARMUP_ITERS):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()

            d_batch = make_batch(mx, m_mj, B)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(nsteps):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            sps = B * nsteps / elapsed
            results[f"B={B}"] = {"elapsed_s": elapsed, "steps_per_s": sps}
            log.info("    B=%5d: %8.1f ms  (%14s steps/s)", B, elapsed * 1e3, f"{sps:,.0f}")
        except Exception:
            log.info("    B=%5d: FAILED", B)
            traceback.print_exc()
            results[f"B={B}"] = None
    return results


def bench_mjx(m_mj, batch_sizes, nsteps):
    """MJX jit(vmap(step)) across batch sizes."""
    try:
        import jax
        from mujoco import mjx
    except ImportError:
        log.info("  JAX not installed -- skipping MJX comparison.")
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
            log.info("    B=%5d: %8.1f ms  (%14s steps/s)", B, elapsed * 1e3, f"{sps:,.0f}")
        except Exception:
            log.info("    B=%5d: FAILED", B)
            traceback.print_exc()
            results[f"B={B}"] = None
    return results


def log_summary(model_name, all_results, batch_sizes):
    """Log a clean summary table."""
    log.info("")
    log.info("=" * 100)
    log.info("  Summary: %s", model_name)
    log.info("=" * 100)

    configs = list(all_results.keys())
    header = f"{'Config':40s}"
    for B in batch_sizes:
        header += f"  {'B=' + str(B):>14s}"
    log.info(header)
    log.info("-" * len(header))

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
        log.info(row)
    log.info("")


def bench_model(model_name, batch_sizes, nsteps, only):
    """Run all requested benchmarks for a single model."""
    log.info("")
    log.info("#" * 100)
    log.info("#  Model: %s", model_name)
    log.info("#" * 100)

    xml = load_model_xml(model_name)
    m_mj = mujoco.MjModel.from_xml_string(xml)
    mx = mujoco_torch.device_put(m_mj).to(DEVICE)

    model_results = {}

    if only in (None, "compile"):
        log.info("\n  torch compile(fullgraph=True):")
        model_results["torch compile"] = bench_compile(mx, m_mj, batch_sizes, nsteps)

    if only in (None, "c"):
        log.info("\n  MuJoCo C (sequential, B=1):")
        r = bench_mujoco_c(m_mj, nsteps)
        sps = r["B=1"]["steps_per_s"]
        log.info("    B=    1: %8.1f ms  (%14s steps/s)", r["B=1"]["elapsed_s"] * 1e3, f"{sps:,.0f}")
        model_results["MuJoCo C (seq)"] = r

    if only in (None, "loop"):
        log.info("\n  mujoco-torch loop (B=1):")
        r = bench_torch_loop(mx, m_mj, nsteps)
        sps = r["B=1"]["steps_per_s"]
        log.info("    B=    1: %8.1f ms  (%14s steps/s)", r["B=1"]["elapsed_s"] * 1e3, f"{sps:,.0f}")
        model_results["torch loop (seq)"] = r

    if only in (None, "vmap"):
        log.info("\n  mujoco-torch vmap (eager):")
        model_results["torch vmap (eager)"] = bench_vmap(mx, m_mj, batch_sizes, nsteps)

    if only in (None, "mjx"):
        gc.collect()
        torch.cuda.empty_cache()
        log.info("\n  MJX jit(vmap(step)):")
        mjx_r = bench_mjx(m_mj, batch_sizes, nsteps)
        if mjx_r is not None:
            model_results["MJX jit(vmap)"] = mjx_r

    log_summary(model_name, model_results, batch_sizes)
    return model_results


def main():
    args = parse_args()
    models = ALL_MODELS if "all" in args.model else args.model
    output = args.output

    torch.set_default_device(DEVICE)

    all_results = {}
    for model_name in models:
        all_results[model_name] = bench_model(
            model_name, args.batch_sizes, args.nsteps, args.only
        )

    out_path = output or ("bench_all.json" if len(models) > 1 else f"bench_{models[0]}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results written to %s", out_path)


if __name__ == "__main__":
    main()
