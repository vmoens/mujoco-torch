#!/usr/bin/env python3
"""End-to-end comparison: MuJoCo (C) vs MJX (JAX) vs mujoco-torch (PyTorch).

Run from the repo root:
    source .venv/bin/activate
    python examples/e2e_comparison.py
"""

import time

import mujoco
import numpy as np
import torch

# ── Model ────────────────────────────────────────────────────────────────────
# Ant model ships with the repo; you can swap in any MuJoCo XML.
from etils import epath

MODEL_XML = (
    epath.resource_path("mujoco_torch") / "test_data" / "ant.xml"
).read_text()

NSTEPS = 200
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  1.  MuJoCo  (C engine – reference)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  MuJoCo (C)")
print("=" * 72)

m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)
d_mj = mujoco.MjData(m_mj)

# small random kick so dynamics are interesting
d_mj.qvel[:] = 0.01 * np.random.randn(m_mj.nv)

t0 = time.perf_counter()
for _ in range(NSTEPS):
    mujoco.mj_step(m_mj, d_mj)
t_mj = time.perf_counter() - t0

qpos_mj = d_mj.qpos.copy()
qvel_mj = d_mj.qvel.copy()
print(f"  {NSTEPS} steps in {t_mj*1e3:.1f} ms")
print(f"  qpos[:5] = {qpos_mj[:5]}")
print(f"  qvel[:5] = {qvel_mj[:5]}")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  2.  mujoco-torch  (PyTorch port of MJX)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  mujoco-torch (PyTorch)")
print("=" * 72)

import mujoco_torch  # noqa: E402

torch.set_default_dtype(torch.float64)

m_torch = mujoco.MjModel.from_xml_string(MODEL_XML)
d_torch_ref = mujoco.MjData(m_torch)

# same initial kick
np.random.seed(SEED)
d_torch_ref.qvel[:] = 0.01 * np.random.randn(m_torch.nv)

mx = mujoco_torch.device_put(m_torch)
dx = mujoco_torch.device_put(d_torch_ref)

# ── eager ──
dx_eager = mujoco_torch.device_put(d_torch_ref)
# warm-up
mujoco_torch.step(mx, dx_eager)
dx_eager = mujoco_torch.device_put(d_torch_ref)

t0 = time.perf_counter()
for _ in range(NSTEPS):
    dx_eager = mujoco_torch.step(mx, dx_eager)
t_torch_eager = time.perf_counter() - t0

qpos_torch = dx_eager.qpos.detach().cpu().numpy()
qvel_torch = dx_eager.qvel.detach().cpu().numpy()
print(f"  eager:    {NSTEPS} steps in {t_torch_eager*1e3:.1f} ms")
print(f"  qpos[:5] = {qpos_torch[:5]}")
print(f"  qvel[:5] = {qvel_torch[:5]}")

# ── compiled ──
print()
print("  Compiling with torch.compile ...")
step_compiled = torch.compile(mujoco_torch.step)
dx_comp = mujoco_torch.device_put(d_torch_ref)
# warm-up / compile — run a few steps so all shape variants are cached
print("  (warm-up trace + compile …)")
t_compile_start = time.perf_counter()
for _ in range(3):
    dx_comp = step_compiled(mx, dx_comp)
t_compile = time.perf_counter() - t_compile_start
print(f"  warm-up compile time ({3} steps): {t_compile:.1f} s")

dx_comp = mujoco_torch.device_put(d_torch_ref)
t0 = time.perf_counter()
for _ in range(NSTEPS):
    dx_comp = step_compiled(mx, dx_comp)
t_torch_compiled = time.perf_counter() - t0

qpos_torch_c = dx_comp.qpos.detach().cpu().numpy()
qvel_torch_c = dx_comp.qvel.detach().cpu().numpy()
print(f"  compiled: {NSTEPS} steps in {t_torch_compiled*1e3:.1f} ms")
print(f"  qpos[:5] = {qpos_torch_c[:5]}")
print(f"  qvel[:5] = {qvel_torch_c[:5]}")
print(f"  compiled vs eager max|Δqpos| = {np.abs(qpos_torch_c - qpos_torch).max():.2e}")
print()

# ─────────────────────────────────────────────────────────────────────────────
#  3.  MJX  (JAX – official XLA port)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  MJX (JAX)")
print("=" * 72)

try:
    import jax
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)

    m_jax = mujoco.MjModel.from_xml_string(MODEL_XML)
    d_jax_ref = mujoco.MjData(m_jax)
    np.random.seed(SEED)
    d_jax_ref.qvel[:] = 0.01 * np.random.randn(m_jax.nv)

    mx_jax = mjx.put_model(m_jax)
    dx_jax = mjx.put_data(m_jax, d_jax_ref)

    step_jax = jax.jit(mjx.step)

    # warm-up / compile
    dx_warm_jax = step_jax(mx_jax, dx_jax)
    jax.block_until_ready(dx_warm_jax.qpos)

    # reset
    dx_jax = mjx.put_data(m_jax, d_jax_ref)

    t0 = time.perf_counter()
    for _ in range(NSTEPS):
        dx_jax = step_jax(mx_jax, dx_jax)
    jax.block_until_ready(dx_jax.qpos)
    t_jax = time.perf_counter() - t0

    qpos_jax = np.array(dx_jax.qpos)
    qvel_jax = np.array(dx_jax.qvel)
    print(f"  {NSTEPS} steps in {t_jax*1e3:.1f} ms")
    print(f"  qpos[:5] = {qpos_jax[:5]}")
    print(f"  qvel[:5] = {qvel_jax[:5]}")
    has_jax = True

except ImportError:
    print("  JAX not installed – skipping MJX comparison.")
    print("  Install with:  pip install jax jaxlib")
    has_jax = False

print()

# ─────────────────────────────────────────────────────────────────────────────
#  4.  Numerical comparison
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  Comparison")
print("=" * 72)


def _compare(name, a, b):
    diff = np.abs(a - b)
    print(f"  {name:30s}  max|Δ|={diff.max():.6e}  mean|Δ|={diff.mean():.6e}")


_compare("torch vs MuJoCo  qpos", qpos_torch, qpos_mj)
_compare("torch vs MuJoCo  qvel", qvel_torch, qvel_mj)
if has_jax:
    _compare("JAX   vs MuJoCo  qpos", qpos_jax, qpos_mj)
    _compare("JAX   vs MuJoCo  qvel", qvel_jax, qvel_mj)
    _compare("torch vs JAX     qpos", qpos_torch, qpos_jax)
    _compare("torch vs JAX     qvel", qvel_torch, qvel_jax)
print()

# ─────────────────────────────────────────────────────────────────────────────
#  5.  Timing summary
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("  Timing  (wall-clock, single-threaded CPU)")
print("=" * 72)
print(f"  {'MuJoCo (C)':30s}  {t_mj*1e3:8.1f} ms  ({NSTEPS/t_mj:,.0f} steps/s)")
print(f"  {'mujoco-torch eager':30s}  {t_torch_eager*1e3:8.1f} ms  ({NSTEPS/t_torch_eager:,.0f} steps/s)")
print(f"  {'mujoco-torch compiled':30s}  {t_torch_compiled*1e3:8.1f} ms  ({NSTEPS/t_torch_compiled:,.0f} steps/s)")
if has_jax:
    print(f"  {'MJX (JAX jit)':30s}  {t_jax*1e3:8.1f} ms  ({NSTEPS/t_jax:,.0f} steps/s)")
print()
speedup = t_torch_eager / t_torch_compiled if t_torch_compiled > 0 else float('inf')
print(f"  torch.compile speedup vs eager: {speedup:.1f}x")
