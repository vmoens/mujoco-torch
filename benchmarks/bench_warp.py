"""Benchmark: MuJoCo Warp graph-captured step across models and batch sizes."""

import mujoco
import numpy as np
import pytest

from benchmarks._helpers import NSTEPS, ROUNDS, SEED, WARMUP_ROUNDS, load_model


@pytest.mark.warp
def test_warp(benchmark, model_name, batch_size):
    import mujoco_warp as mjw
    import warp as wp

    wp.init()

    m_mj = load_model(model_name)
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = 0.01 * np.random.RandomState(SEED).randn(m_mj.nv)

    m = mjw.put_model(m_mj)
    d = mjw.put_data(m_mj, d_mj, nworld=batch_size)

    # Warm-up: ensures kernels are JIT-compiled before we capture the graph.
    mjw.step(m, d)
    wp.synchronize()

    with wp.ScopedCapture() as capture:
        mjw.step(m, d)
    graph = capture.graph

    def run():
        for _ in range(NSTEPS):
            wp.capture_launch(graph)
        wp.synchronize()

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = batch_size * NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=batch_size,
        backend="MuJoCo Warp",
        steps_per_s=sps,
    )
