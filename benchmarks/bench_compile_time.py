"""Benchmark: cold compile + first call across backends.

For PyTorch ``torch.compile`` there is no separate "compile backward" step —
the backward graph is traced lazily on the first ``.backward()`` call.  To
compare apples-to-apples with MJX (where ``jax.jit(jax.grad(...))`` produces a
single fwd+bwd HLO), we measure two regions per backend:

* **fwd** — cold state → wrap with the backend's compile mechanism → first
  forward call → synchronize.
* **fwd+bwd** — same, but also drive a ``.backward()`` before stopping the
  timer.

Backward-compile cost can then be read as ``(fwd+bwd) - fwd`` per backend.

Each test does a single cold timing (rounds=1, no warmup); ``setup`` builds
the model/batch outside the timed region so only the compile + first call is
measured.
"""

import os
import tempfile

import mujoco
import numpy as np
import pytest
import torch
import torch._inductor.config as inductor_config

import mujoco_torch
from benchmarks._helpers import DEVICE, SEED, load_model, make_batch, warm_caches


def _force_cold_inductor():
    """Force a cold Inductor compile by redirecting the on-disk cache dir.

    Note: we deliberately do NOT set ``force_disable_caches = True`` — under
    torch >= 2.13 nightlies it triggers a TypeError("unhashable type:
    numpy.ndarray") during dynamo's guard generation. A fresh tempdir for
    ``TORCHINDUCTOR_CACHE_DIR`` plus ``_dynamo.reset()`` is enough to make
    each test cold-compile.
    """
    tmp = tempfile.mkdtemp(prefix="inductor_cold_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tmp
    torch._dynamo.reset()
    torch.compiler.reset()


def _emit(benchmark, *, model_name, batch_size, backend):
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend=backend,
        compile_time_s=benchmark.stats.stats.mean,
    )


# ---------------------------------------------------------------------------
# mujoco-torch
# ---------------------------------------------------------------------------


def _torch_setup(model_name, batch_size, *, requires_grad):
    _force_cold_inductor()
    m_mj = load_model(model_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)
    warm_caches(mx, m_mj, DEVICE)
    d_batch = make_batch(mx, m_mj, batch_size, DEVICE)
    if requires_grad:
        d_batch.qvel.requires_grad_(True)
    return (mx, d_batch), {}


def test_torch_compile_fwd(benchmark, model_name, batch_size):
    def measure(mx, d_batch):
        compiled = torch.compile(torch.vmap(lambda d: mujoco_torch.step(mx, d)), fullgraph=True)
        out = compiled(d_batch)
        _ = out.qpos.sum().item()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    benchmark.pedantic(
        measure,
        setup=lambda: _torch_setup(model_name, batch_size, requires_grad=False),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _emit(benchmark, model_name=model_name, batch_size=batch_size,
          backend="torch compile (cold fwd)")


def test_torch_compile_fwd_bwd(benchmark, model_name, batch_size):
    def measure(mx, d_batch):
        compiled = torch.compile(torch.vmap(lambda d: mujoco_torch.step(mx, d)), fullgraph=True)
        out = compiled(d_batch)
        out.qpos.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    benchmark.pedantic(
        measure,
        setup=lambda: _torch_setup(model_name, batch_size, requires_grad=True),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _emit(benchmark, model_name=model_name, batch_size=batch_size,
          backend="torch compile (cold fwd+bwd)")


# ---------------------------------------------------------------------------
# MJX
# ---------------------------------------------------------------------------


def _mjx_setup(model_name, batch_size):
    import jax
    from mujoco import mjx

    jax.config.update("jax_enable_x64", True)
    jax.clear_caches()

    m_mj = load_model(model_name)
    mx_jax = mjx.put_model(m_mj)

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
    return (jax, mjx, mx_jax, dx_jax, jax.numpy.array(qvels)), {}


@pytest.mark.mjx
def test_mjx_compile_fwd(benchmark, model_name, batch_size):
    def measure(jax, mjx, mx_jax, dx_jax, _qvels):
        step_fn = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        out = step_fn(mx_jax, dx_jax)
        jax.block_until_ready(out.qpos)

    benchmark.pedantic(
        measure,
        setup=lambda: _mjx_setup(model_name, batch_size),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _emit(benchmark, model_name=model_name, batch_size=batch_size,
          backend="MJX (cold fwd)")


@pytest.mark.mjx
def test_mjx_compile_fwd_bwd(benchmark, model_name, batch_size):
    def measure(jax, mjx, mx_jax, dx_jax, qvels_jax):
        def loss_fn(qvel):
            d = dx_jax.replace(qvel=qvel)
            return jax.vmap(mjx.step, in_axes=(None, 0))(mx_jax, d).qpos.sum()

        grad_fn = jax.jit(jax.grad(loss_fn))
        g = grad_fn(qvels_jax)
        jax.block_until_ready(g)

    benchmark.pedantic(
        measure,
        setup=lambda: _mjx_setup(model_name, batch_size),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _emit(benchmark, model_name=model_name, batch_size=batch_size,
          backend="MJX (cold fwd+bwd)")


# ---------------------------------------------------------------------------
# MuJoCo Warp (forward only: mujoco_warp.step does not bridge to torch.autograd)
# ---------------------------------------------------------------------------


def _warp_setup(model_name, batch_size):
    import mujoco_warp as mjw  # noqa: F401  (ensure available before timing)
    import warp as wp

    wp.init()
    try:
        wp.build.clear_kernel_cache()
    except Exception:
        pass

    m_mj = load_model(model_name)
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = 0.01 * np.random.RandomState(SEED).randn(m_mj.nv)
    return (m_mj, d_mj), {}


@pytest.mark.warp
def test_warp_compile_fwd(benchmark, model_name, batch_size):
    def measure(m_mj, d_mj):
        import mujoco_warp as mjw
        import warp as wp

        m = mjw.put_model(m_mj)
        d = mjw.put_data(m_mj, d_mj, nworld=batch_size)
        mjw.step(m, d)  # JIT-compiles kernels
        with wp.ScopedCapture() as capture:
            mjw.step(m, d)
        wp.capture_launch(capture.graph)
        wp.synchronize()

    benchmark.pedantic(
        measure,
        setup=lambda: _warp_setup(model_name, batch_size),
        rounds=1,
        iterations=1,
        warmup_rounds=0,
    )
    _emit(benchmark, model_name=model_name, batch_size=batch_size,
          backend="MuJoCo Warp (cold fwd)")
