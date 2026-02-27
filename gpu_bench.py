"""GPU benchmark for mujoco-torch vs MuJoCo C vs MJX (JAX).

Tests compile modes: default, reduce-overhead (CUDA graphs via compile),
max-autotune, and different backends.
"""

import sys
import time
import traceback

import mujoco
import numpy as np
import torch
from etils import epath

import mujoco_torch

torch.set_default_dtype(torch.float64)

DEVICE = "cuda"
MODEL_XML = (
    epath.resource_path("mujoco_torch") / "test_data" / "humanoid.xml"
).read_text()
BATCH_SIZES = [1, 16, 64, 256, 1024]
NSTEPS = 100
SEED = 42


def make_batched_qvel(nv, batch_size, seed):
    rng = np.random.RandomState(seed)
    return 0.01 * rng.randn(batch_size, nv)


m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)

# ── MuJoCo C (CPU baseline) ──
print("=" * 80)
print("  MuJoCo (C) — sequential over batch")
print("=" * 80, flush=True)
mj_results = {}
for B in BATCH_SIZES:
    qvels = make_batched_qvel(m_mj.nv, B, SEED)
    ds = [mujoco.MjData(m_mj) for _ in range(B)]
    for i, d in enumerate(ds):
        d.qvel[:] = qvels[i]
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        for d in ds:
            mujoco.mj_step(m_mj, d)
    t = time.perf_counter() - t0
    mj_results[B] = t
    print(f"  B={B:4d}:  {t * 1e3:8.1f} ms  ({B * NSTEPS / t:,.0f} steps/s)", flush=True)
print(flush=True)

# device_put on CPU, then move to CUDA
mx = mujoco_torch.device_put(m_mj).to(DEVICE)
torch.set_default_device(DEVICE)
print("Model tensor device:", mx.body_pos.device, flush=True)


def _make_batch(mx, m_mj, B, seed):
    qvels = make_batched_qvel(m_mj.nv, B, seed)
    envs = []
    with torch.device("cpu"):
        for i in range(B):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d).to(DEVICE))
    return torch.stack(envs, dim=0)


# ── vmap (CUDA) — eager baseline ──
print("=" * 80)
print("  mujoco-torch vmap (CUDA) — eager, no compile")
print("=" * 80, flush=True)
vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
vmap_results = {}
for B in BATCH_SIZES:
    d_batch = _make_batch(mx, m_mj, B, SEED)
    d_batch = vmap_step(d_batch)
    torch.cuda.synchronize()

    d_batch = _make_batch(mx, m_mj, B, SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = vmap_step(d_batch)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    vmap_results[B] = t
    print(f"  B={B:4d}:  {t * 1e3:8.1f} ms  ({B * NSTEPS / t:,.0f} steps/s)", flush=True)
print(flush=True)


def _bench_compile(label, compile_kwargs, batch_sizes=BATCH_SIZES, warmup_iters=5):
    """Benchmark a compile configuration across batch sizes."""
    print("=" * 80)
    print(f"  {label}")
    print("=" * 80, flush=True)
    results = {}
    for B in batch_sizes:
        compiled_fn = torch.compile(vmap_step, **compile_kwargs)
        d_batch = _make_batch(mx, m_mj, B, SEED)
        print(f"  B={B:4d}: compiling...", end="", flush=True)
        try:
            for _ in range(warmup_iters):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()
            print(" timing...", end="", flush=True)

            d_batch = _make_batch(mx, m_mj, B, SEED)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(NSTEPS):
                d_batch = compiled_fn(d_batch)
            torch.cuda.synchronize()
            t = time.perf_counter() - t0
            results[B] = t
            print(f"  {t * 1e3:8.1f} ms  ({B * NSTEPS / t:,.0f} steps/s)", flush=True)
        except Exception:
            print(f"  FAILED", flush=True)
            traceback.print_exc()
            results[B] = None
    print(flush=True)
    return results


# ── compile modes ──
compile_configs = [
    ("compile(default)", {}),
    ("compile(fullgraph)", {"fullgraph": True}),
    ("compile(reduce-overhead) — CUDA graphs via compile", {"mode": "reduce-overhead"}),
    ("compile(reduce-overhead+fullgraph)", {"mode": "reduce-overhead", "fullgraph": True}),
    ("compile(max-autotune)", {"mode": "max-autotune"}),
    ("compile(max-autotune+fullgraph)", {"mode": "max-autotune", "fullgraph": True}),
]

all_compile_results = {}
for label, kwargs in compile_configs:
    all_compile_results[label] = _bench_compile(label, kwargs)

# ── MJX (JAX) ──
print("=" * 80)
print("  MJX (JAX) — jit(vmap(step)) = true batch parallelism")
print("=" * 80, flush=True)
has_jax = False
jax_results = {}
try:
    import jax
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    m_jax = mujoco.MjModel.from_xml_string(MODEL_XML)
    mx_jax = mjx.put_model(m_jax)
    step_jax_vmap = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    for B in BATCH_SIZES:
        qvels = make_batched_qvel(m_jax.nv, B, SEED)
        d_ref = mujoco.MjData(m_jax)
        dx_single = mjx.put_data(m_jax, d_ref)

        def _tile_leaf(x):
            if not hasattr(x, "ndim"):
                return x
            if x.ndim == 0:
                return jax.numpy.broadcast_to(x, (B,))
            return jax.numpy.tile(x, (B,) + (1,) * x.ndim)

        dx_jax = jax.tree.map(_tile_leaf, dx_single)
        dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))
        dx_warm = step_jax_vmap(mx_jax, dx_jax)
        jax.block_until_ready(dx_warm.qpos)

        dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))
        t0 = time.perf_counter()
        for _ in range(NSTEPS):
            dx_jax = step_jax_vmap(mx_jax, dx_jax)
        jax.block_until_ready(dx_jax.qpos)
        t = time.perf_counter() - t0
        jax_results[B] = t
        print(
            f"  B={B:4d}:  {t * 1e3:8.1f} ms  ({B * NSTEPS / t:,.0f} steps/s)",
            flush=True,
        )
    has_jax = True
except ImportError:
    print("  JAX not installed — skipping MJX comparison.", flush=True)
print(flush=True)

# ── Summary ──
print("=" * 80)
print("  Summary  (steps/s = batch_size x nsteps / time)")
print("=" * 80)

sps_fmt = lambda t, B: f"{B * NSTEPS / t:>16,.0f}" if t else f"{'FAIL':>16s}"

cols = ["B", "MuJoCo(C)", "vmap"]
compile_labels = list(all_compile_results.keys())
for lbl in compile_labels:
    short = lbl.split("(")[1].rstrip(")").split(" —")[0]
    cols.append(short)
if has_jax:
    cols.append("MJX(JAX)")
header = "  ".join(f"{c:>16s}" for c in cols)
print(header)
print("-" * len(header))

for B in BATCH_SIZES:
    row = f"  {B:>14d}  {sps_fmt(mj_results[B], B)}  {sps_fmt(vmap_results[B], B)}"
    for lbl in compile_labels:
        row += f"  {sps_fmt(all_compile_results[lbl].get(B), B)}"
    if has_jax and B in jax_results:
        row += f"  {sps_fmt(jax_results[B], B)}"
    print(row)
print(flush=True)
