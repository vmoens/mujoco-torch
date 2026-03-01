"""Benchmark: mujoco-torch compile(fullgraph=True) across models and batch sizes."""

import torch

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


def test_compile(benchmark, model_name, batch_size):
    m_mj = load_model(model_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)
    warm_caches(mx, m_mj, DEVICE)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
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
        backend="torch compile",
        steps_per_s=sps,
    )
