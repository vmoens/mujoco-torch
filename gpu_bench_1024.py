"""Focused B=1024 GPU benchmark — one compile config at a time, cache reset.

Tests whether compile cache accumulation from previous configs
affects performance. Each compile config gets a fresh torch._dynamo reset.
"""

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
B = 1024
NSTEPS = 1000
WARMUP = 10
SEED = 42


def make_batched_qvel(nv, batch_size, seed):
    rng = np.random.RandomState(seed)
    return 0.01 * rng.randn(batch_size, nv)


m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)

# ── MuJoCo C at B=1 (pessimistic sequential baseline) ──
print("=" * 80)
print("  MuJoCo (C) — B=1 sequential baseline")
print("=" * 80, flush=True)
d_mj = mujoco.MjData(m_mj)
d_mj.qvel[:] = make_batched_qvel(m_mj.nv, 1, SEED)[0]
t0 = time.perf_counter()
for _ in range(NSTEPS):
    mujoco.mj_step(m_mj, d_mj)
t_c = time.perf_counter() - t0
print(f"  B=1: {t_c * 1e3:.1f} ms  ({NSTEPS / t_c:,.0f} steps/s)", flush=True)
print(flush=True)

# ── Setup ──
mx = mujoco_torch.device_put(m_mj).to(DEVICE)
torch.set_default_device(DEVICE)
print(f"Model device: {mx.body_pos.device}", flush=True)


def _make_batch(seed=SEED):
    qvels = make_batched_qvel(m_mj.nv, B, seed)
    envs = []
    with torch.device("cpu"):
        for i in range(B):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d).to(DEVICE))
    return torch.stack(envs, dim=0)


def _bench(label, fn, warmup=WARMUP):
    """Run benchmark with warmup and timing."""
    d_batch = _make_batch()
    print(f"  {label}: warming up ({warmup} iters)...", end="", flush=True)
    for _ in range(warmup):
        d_batch = fn(d_batch)
    torch.cuda.synchronize()
    print(f" timing ({NSTEPS} steps)...", end="", flush=True)

    d_batch = _make_batch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = fn(d_batch)
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
    sps = B * NSTEPS / t
    print(f"  {t * 1e3:.1f} ms  ({sps:,.0f} steps/s)", flush=True)
    return t, sps


results = {}

# ── vmap (CUDA) — eager baseline ──
print("=" * 80)
print(f"  vmap eager — B={B}")
print("=" * 80, flush=True)
vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
t, sps = _bench("vmap", vmap_step)
results["vmap"] = sps

# ── Compile configs (each with fresh cache) ──
compile_configs = [
    ("compile(default)", {}),
    ("compile(fullgraph)", {"fullgraph": True}),
    ("compile(reduce-overhead)", {"mode": "reduce-overhead"}),
    ("compile(reduce-overhead+fullgraph)", {"mode": "reduce-overhead", "fullgraph": True}),
]

for label, kwargs in compile_configs:
    torch._dynamo.reset()
    print("=" * 80)
    print(f"  {label} — B={B}")
    print("=" * 80, flush=True)
    compiled_fn = torch.compile(vmap_step, **kwargs)
    try:
        t, sps = _bench(label, compiled_fn)
        results[label] = sps
    except Exception:
        print("  FAILED", flush=True)
        traceback.print_exc()
        results[label] = None

# ── Summary ──
print()
print("=" * 80)
print(f"  Summary — B={B}, {NSTEPS} steps")
print("=" * 80)
print(f"  MuJoCo C (B=1):  {NSTEPS / t_c:>12,.0f} steps/s")
for label, sps in results.items():
    val = f"{sps:>12,.0f}" if sps else "        FAIL"
    print(f"  {label:40s} {val} steps/s")
print(flush=True)
