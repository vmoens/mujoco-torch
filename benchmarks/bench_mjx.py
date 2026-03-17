"""Benchmark: MJX jit(vmap(step)) across models and batch sizes."""

import mujoco
import numpy as np
import pytest

from benchmarks._helpers import NSTEPS, ROUNDS, SEED, WARMUP_ROUNDS, load_model


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

    dx_warm = step_fn(mx_jax, dx_jax)
    jax.block_until_ready(dx_warm.qpos)

    def run():
        dx = dx_jax.replace(qvel=jax.numpy.array(qvels))
        for _ in range(NSTEPS):
            dx = step_fn(mx_jax, dx)
        jax.block_until_ready(dx.qpos)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = batch_size * NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend="MJX jit(vmap)",
        steps_per_s=sps,
    )
