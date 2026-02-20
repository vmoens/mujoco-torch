"""Benchmark suite for mujoco-torch step() across backends."""

import mujoco
import numpy as np
import pytest
import torch

import mujoco_torch
from benchmarks._helpers import NSTEPS, SEED, load_model, make_batch, make_single

ROUNDS = 5
WARMUP_ROUNDS = 2


# ── mujoco-torch  vmap ──────────────────────────────────────────────────────


def test_mujoco_torch_vmap(benchmark, model_name, batch_size, device):
    torch.set_default_dtype(torch.float64)
    m_mj = load_model(model_name)
    mx = mujoco_torch.device_put(m_mj)
    if device != "cpu":
        mx = mx.to(device)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

    d_batch = make_batch(mx, m_mj, batch_size, device)
    # prime vmap trace once
    vmap_step(d_batch)

    d_batch = make_batch(mx, m_mj, batch_size, device)

    def run():
        nonlocal d_batch
        for _ in range(NSTEPS):
            d_batch = vmap_step(d_batch)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend="mujoco_torch_vmap",
        device=device,
    )


# ── mujoco-torch  loop ──────────────────────────────────────────────────────


def test_mujoco_torch_loop(benchmark, model_name, device):
    torch.set_default_dtype(torch.float64)
    m_mj = load_model(model_name)
    mx = mujoco_torch.device_put(m_mj)
    if device != "cpu":
        mx = mx.to(device)

    dx = make_single(m_mj, device)
    # warmup
    mujoco_torch.step(mx, dx)

    def run():
        dx_local = make_single(m_mj, device)
        for _ in range(NSTEPS):
            dx_local = mujoco_torch.step(mx, dx_local)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info.update(
        model=model_name,
        batch_size=1,
        backend="mujoco_torch_loop",
        device=device,
    )


# ── MuJoCo C  loop ──────────────────────────────────────────────────────────


def test_mujoco_c_loop(benchmark, model_name):
    m_mj = load_model(model_name)

    def run():
        d = mujoco.MjData(m_mj)
        rng = np.random.RandomState(SEED)
        d.qvel[:] = 0.01 * rng.randn(m_mj.nv)
        for _ in range(NSTEPS):
            mujoco.mj_step(m_mj, d)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info.update(
        model=model_name,
        batch_size=1,
        backend="mujoco_c_loop",
        device="cpu",
    )


# ── MJX  vmap ───────────────────────────────────────────────────────────────


@pytest.mark.mjx
def test_mjx_vmap(benchmark, model_name, batch_size):
    import jax
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)

    m_mj = load_model(model_name)
    mx_jax = mjx.put_model(m_mj)
    step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

    rng = np.random.RandomState(SEED)
    qvels = 0.01 * rng.randn(batch_size, m_mj.nv)

    d_ref = mujoco.MjData(m_mj)
    dx_single = mjx.put_data(m_mj, d_ref)

    def _tile_leaf(x):
        if not hasattr(x, "ndim"):
            return x
        if x.ndim == 0:
            return jax.numpy.broadcast_to(x, (batch_size,))
        return jax.numpy.tile(x, (batch_size,) + (1,) * x.ndim)

    dx_jax = jax.tree.map(_tile_leaf, dx_single)
    dx_jax = dx_jax.replace(qvel=jax.numpy.array(qvels))

    # warmup / compile
    dx_warm = step_fn(mx_jax, dx_jax)
    jax.block_until_ready(dx_warm.qpos)

    def run():
        dx = dx_jax.replace(qvel=jax.numpy.array(qvels))
        for _ in range(NSTEPS):
            dx = step_fn(mx_jax, dx)
        jax.block_until_ready(dx.qpos)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend="mjx_vmap",
        device="cpu",
    )


# ── MJX  loop ───────────────────────────────────────────────────────────────


@pytest.mark.mjx
def test_mjx_loop(benchmark, model_name):
    import jax
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)

    m_mj = load_model(model_name)
    mx_jax = mjx.put_model(m_mj)
    step_fn = jax.jit(mjx.step)

    d_ref = mujoco.MjData(m_mj)
    rng = np.random.RandomState(SEED)
    d_ref.qvel[:] = 0.01 * rng.randn(m_mj.nv)
    dx_jax = mjx.put_data(m_mj, d_ref)

    # warmup / compile
    dx_warm = step_fn(mx_jax, dx_jax)
    jax.block_until_ready(dx_warm.qpos)

    def run():
        dx = mjx.put_data(m_mj, d_ref)
        for _ in range(NSTEPS):
            dx = step_fn(mx_jax, dx)
        jax.block_until_ready(dx.qpos)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    benchmark.extra_info.update(
        model=model_name,
        batch_size=1,
        backend="mjx_loop",
        device="cpu",
    )
