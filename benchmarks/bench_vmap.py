"""Benchmark: mujoco-torch vmap (eager) across models and batch sizes."""

import torch

import mujoco_torch
from benchmarks._helpers import DEVICE, NSTEPS, ROUNDS, WARMUP_ROUNDS, load_model, make_batch


def test_vmap(benchmark, model_name, batch_size):
    m_mj = load_model(model_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

    d_batch = make_batch(mx, m_mj, batch_size, DEVICE)
    vmap_step(d_batch)
    torch.cuda.synchronize()
    d_batch = make_batch(mx, m_mj, batch_size, DEVICE)

    def run():
        nonlocal d_batch
        for _ in range(NSTEPS):
            d_batch = vmap_step(d_batch)
        torch.cuda.synchronize()

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = batch_size * NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend="torch vmap (eager)",
        steps_per_s=sps,
    )
