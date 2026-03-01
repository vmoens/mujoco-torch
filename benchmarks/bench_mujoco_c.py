"""Benchmark: MuJoCo C sequential step at B=1."""

import mujoco
import numpy as np

from benchmarks._helpers import NSTEPS, ROUNDS, SEED, WARMUP_ROUNDS, load_model


def test_mujoco_c(benchmark, model_name):
    m_mj = load_model(model_name)

    d = mujoco.MjData(m_mj)
    d.qvel[:] = 0.01 * np.random.RandomState(SEED).randn(m_mj.nv)
    for _ in range(10):
        mujoco.mj_step(m_mj, d)

    def run():
        d = mujoco.MjData(m_mj)
        d.qvel[:] = 0.01 * np.random.RandomState(SEED).randn(m_mj.nv)
        for _ in range(NSTEPS):
            mujoco.mj_step(m_mj, d)

    benchmark.pedantic(run, rounds=ROUNDS, warmup_rounds=WARMUP_ROUNDS)
    sps = NSTEPS / benchmark.stats.stats.mean
    benchmark.extra_info.update(
        model=model_name,
        batch_size=1,
        backend="MuJoCo C (seq)",
        steps_per_s=sps,
    )
