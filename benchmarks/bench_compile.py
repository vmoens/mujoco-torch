"""Benchmark: mujoco-torch compile(fullgraph=True)."""

import torch
import torch._inductor.config as inductor_config

import mujoco_torch
from benchmarks._helpers import (
    COMPILE_WARMUP_ITERS,
    DEVICE,
    NSTEPS,
    ROUNDS,
    WARMUP_ROUNDS,
    load_model,
    make_batch,
    warm_caches,
)


def _bench_compile(benchmark, model_name, batch_size, *, backend_label,
                   inductor_tuning=False, step_kwargs=None):
    inductor_config.coordinate_descent_tuning = inductor_tuning
    inductor_config.aggressive_fusion = inductor_tuning

    m_mj = load_model(model_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)
    warm_caches(mx, m_mj, DEVICE)

    kw = step_kwargs or {}
    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d, **kw))
    compiled_fn = torch.compile(vmap_step, fullgraph=True)
    d_batch = make_batch(mx, m_mj, batch_size, DEVICE)

    for _ in range(COMPILE_WARMUP_ITERS):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()
    d_batch = make_batch(mx, m_mj, batch_size, DEVICE)

    def run():
        nonlocal d_batch
        for _ in range(NSTEPS):
            d_batch = compiled_fn(d_batch)
        torch.cuda.synchronize()

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = batch_size * NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend=backend_label,
        steps_per_s=sps,
    )


def test_compile(benchmark, model_name, batch_size):
    _bench_compile(
        benchmark, model_name, batch_size,
        backend_label="torch compile",
    )


def test_compile_h4(benchmark, model_name, batch_size):
    _bench_compile(
        benchmark, model_name, batch_size,
        backend_label="torch compile (H4)",
        inductor_tuning=True,
    )


def test_compile_h4_h8(benchmark, model_name, batch_size):
    _bench_compile(
        benchmark, model_name, batch_size,
        backend_label="torch compile (H4+H8)",
        inductor_tuning=True,
        step_kwargs={"fixed_iterations": True},
    )
