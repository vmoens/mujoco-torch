"""Benchmark: mujoco-torch sequential loop at B=1."""

import torch

import mujoco_torch
from benchmarks._helpers import DEVICE, NSTEPS, ROUNDS, WARMUP_ROUNDS, load_model, make_single


def test_loop(benchmark, model_name):
    m_mj = load_model(model_name)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)

    dx = make_single(m_mj, DEVICE)
    for _ in range(5):
        dx = mujoco_torch.step(mx, dx)
    torch.cuda.synchronize()

    def run():
        nonlocal dx
        dx = make_single(m_mj, DEVICE)
        torch.cuda.synchronize()
        for _ in range(NSTEPS):
            dx = mujoco_torch.step(mx, dx)
        torch.cuda.synchronize()

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=1,
        backend="torch loop (seq)",
        steps_per_s=sps,
    )
