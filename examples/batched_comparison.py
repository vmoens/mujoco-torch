#!/usr/bin/env python3
"""Batched simulation comparison: MuJoCo (C) vs MJX (JAX) vs mujoco-torch.

Runs B independent simulations for N steps each.

- MuJoCo C: sequential loop over batch (no parallelism)
- mujoco-torch eager (seq): sequential loop over batch (Python overhead per env)
- mujoco-torch vmap: torch.vmap(step) — true batch parallelism
- mujoco-torch compile+vmap: torch.compile(vmap(step))
- MJX (JAX): jit(vmap(step)) — true batch parallelism

Run from the repo root:
    source .venv/bin/activate
    python examples/batched_comparison.py
"""

import time

import mujoco
import numpy as np
import torch

from etils import epath

MODEL_XML = (
    epath.resource_path("mujoco_torch") / "test_data" / "humanoid.xml"
).read_text()

BATCH_SIZES = [1, 4, 16, 64]
NSTEPS = 20
SEED = 42


def make_batched_qvel(nv, batch_size, seed):
    rng = np.random.RandomState(seed)
    return 0.01 * rng.randn(batch_size, nv)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  MuJoCo  (C engine – sequential loop over batch)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  MuJoCo (C) — sequential over batch")
print("=" * 80)

m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)
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

    total_steps = NSTEPS * B
    mj_results[B] = t
    print(f"  B={B:4d}:  {t*1e3:8.1f} ms  "
          f"({total_steps/t:,.0f} steps/s)")

print()

# ─────────────────────────────────────────────────────────────────────────────
#  2.  mujoco-torch  (eager sequential + vmap + compile+vmap)
# ─────────────────────────────────────────────────────────────────────────────

import mujoco_torch

torch.set_default_dtype(torch.float64)

m_torch = mujoco.MjModel.from_xml_string(MODEL_XML)
mx = mujoco_torch.device_put(m_torch)


def _make_batch(mx, m_mj, B, seed):
    """Create a batched Data instance (batch dim 0) with different qvel."""
    qvels = make_batched_qvel(m_mj.nv, B, seed)
    envs = []
    for i in range(B):
        d = mujoco.MjData(m_mj)
        d.qvel[:] = qvels[i]
        envs.append(mujoco_torch.device_put(d))
    return torch.stack(envs, dim=0)


def _make_envs_list(mx, m_mj, B, seed):
    """Create B separate Data instances with different qvel."""
    qvels = make_batched_qvel(m_mj.nv, B, seed)
    envs = []
    for i in range(B):
        d = mujoco.MjData(m_mj)
        d.qvel[:] = qvels[i]
        envs.append(mujoco_torch.device_put(d))
    return envs


# ── 2a. eager sequential ──
print("=" * 80)
print("  mujoco-torch eager — sequential over batch")
print("=" * 80)

torch_eager_results = {}

for B in BATCH_SIZES:
    envs = _make_envs_list(mx, m_mj, B, SEED)
    # warm-up
    mujoco_torch.step(mx, envs[0])

    envs = _make_envs_list(mx, m_mj, B, SEED)
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        for i in range(B):
            envs[i] = mujoco_torch.step(mx, envs[i])
    t = time.perf_counter() - t0

    total_steps = NSTEPS * B
    torch_eager_results[B] = t
    print(f"  B={B:4d}:  {t*1e3:8.1f} ms  "
          f"({total_steps/t:,.0f} steps/s)")

print()

# ── 2b. vmap (eager) ──
print("=" * 80)
print("  mujoco-torch vmap — true batch parallelism")
print("=" * 80)

vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

torch_vmap_results = {}

for B in BATCH_SIZES:
    d_batch = _make_batch(mx, m_mj, B, SEED)
    # warm-up
    vmap_step(d_batch)

    d_batch = _make_batch(mx, m_mj, B, SEED)
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = vmap_step(d_batch)
    t = time.perf_counter() - t0

    total_steps = NSTEPS * B
    torch_vmap_results[B] = t
    print(f"  B={B:4d}:  {t*1e3:8.1f} ms  "
          f"({total_steps/t:,.0f} steps/s)")

print()

# ── 2c. compile + vmap ──
print("=" * 80)
print("  mujoco-torch compile+vmap — compiled batch parallelism")
print("=" * 80)

compiled_vmap_step = torch.compile(vmap_step)

torch_cvmap_results = {}

for B in BATCH_SIZES:
    d_batch = _make_batch(mx, m_mj, B, SEED)
    # warm-up / compile
    for _ in range(5):
        d_batch = compiled_vmap_step(d_batch)

    d_batch = _make_batch(mx, m_mj, B, SEED)
    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        d_batch = compiled_vmap_step(d_batch)
    t = time.perf_counter() - t0

    total_steps = NSTEPS * B
    torch_cvmap_results[B] = t
    print(f"  B={B:4d}:  {t*1e3:8.1f} ms  "
          f"({total_steps/t:,.0f} steps/s)")

print()

# ─────────────────────────────────────────────────────────────────────────────
#  3.  MJX  (JAX – jit + vmap = true batch parallelism)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  MJX (JAX) — jit(vmap(step)) = true batch parallelism")
print("=" * 80)

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
            if not hasattr(x, 'ndim'):
                return x
            # Scalars (0-d) need a leading batch dim; higher-rank tensors get tiled.
            if x.ndim == 0:
                return jax.numpy.broadcast_to(x, (B,))
            return jax.numpy.tile(x, (B,) + (1,) * x.ndim)

        dx_jax = jax.tree.map(_tile_leaf, dx_single)
        dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))

        # warm-up / compile
        dx_warm = step_jax_vmap(mx_jax, dx_jax)
        jax.block_until_ready(dx_warm.qpos)

        dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))
        t0 = time.perf_counter()
        for _ in range(NSTEPS):
            dx_jax = step_jax_vmap(mx_jax, dx_jax)
        jax.block_until_ready(dx_jax.qpos)
        t = time.perf_counter() - t0

        total_steps = NSTEPS * B
        jax_results[B] = t
        print(f"  B={B:4d}:  {t*1e3:8.1f} ms  "
              f"({total_steps/t:,.0f} steps/s)")

    has_jax = True

except ImportError:
    print("  JAX not installed – skipping MJX comparison.")

print()

# ─────────────────────────────────────────────────────────────────────────────
#  4.  Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("  Summary  (steps/s = batch_size × nsteps / time)")
print("=" * 80)

cols = ["B", "MuJoCo(C)", "eager(seq)", "vmap", "compile+vmap"]
if has_jax:
    cols.append("MJX(JAX)")
cols.extend(["vmap/eager", "vmap/C"])
if has_jax:
    cols.append("JAX/vmap")

header = "  ".join(f"{c:>14s}" for c in cols)
print(header)
print("-" * len(header))

for B in BATCH_SIZES:
    t_mj = mj_results[B]
    t_te = torch_eager_results[B]
    t_tv = torch_vmap_results[B]
    t_tc = torch_cvmap_results[B]
    sps = lambda t: B * NSTEPS / t

    row = (f"  {B:>12d}"
           f"  {sps(t_mj):>14,.0f}"
           f"  {sps(t_te):>14,.0f}"
           f"  {sps(t_tv):>14,.0f}"
           f"  {sps(t_tc):>14,.0f}")
    if has_jax and B in jax_results:
        t_jx = jax_results[B]
        row += f"  {sps(t_jx):>14,.0f}"
    row += f"  {t_te / t_tv:>14.2f}x"
    row += f"  {t_mj / t_tv:>14.2f}x"
    if has_jax and B in jax_results:
        row += f"  {t_tv / jax_results[B]:>14.2f}x"
    print(row)
