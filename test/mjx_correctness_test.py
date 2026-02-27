# Copyright 2025 Vincent Moens
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Numerical correctness tests comparing mujoco-torch against MJX (JAX)."""

import mujoco
import numpy as np
import pytest
import torch
from mujoco import mjx

import mujoco_torch
from mujoco_torch._src import test_util
from mujoco_torch._src.types import DisableBit

torch.set_default_dtype(torch.float64)

NSTEPS = 100
SEED = 42

SIMPLE_MODEL = "pendula.xml"
COMPLEX_MODEL = "ant.xml"


def _run_mjx(m_mj, qvel_kick, nsteps, disable_constraint=False):
    """Run MJX simulation and return per-step results."""
    import jax

    jax.config.update("jax_enable_x64", True)

    if disable_constraint:
        m_mj = m_mj.__copy__()
        m_mj.opt.disableflags = m_mj.opt.disableflags | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    mx_jax = mjx.put_model(m_mj)
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = qvel_kick
    dx_jax = mjx.put_data(m_mj, d_mj)

    step_fn = mjx.step

    results = []
    for _ in range(nsteps):
        dx_jax = step_fn(mx_jax, dx_jax)
        results.append(
            {
                "qpos": np.array(dx_jax.qpos),
                "qvel": np.array(dx_jax.qvel),
                "act": np.array(dx_jax.act),
                "time": float(dx_jax.time),
            }
        )
    return results


def _run_torch_single(m_mj, qvel_kick, nsteps, disable_constraint=False):
    """Run mujoco-torch single-env simulation and return per-step results."""
    if disable_constraint:
        m_mj = m_mj.__copy__()
        m_mj.opt.disableflags = m_mj.opt.disableflags | int(DisableBit.CONSTRAINT)
    mx = mujoco_torch.device_put(m_mj)
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = qvel_kick
    dx = mujoco_torch.device_put(d_mj)

    results = []
    for _ in range(nsteps):
        dx = mujoco_torch.step(mx, dx)
        results.append(
            {
                "qpos": dx.qpos.numpy(),
                "qvel": dx.qvel.numpy(),
                "act": dx.act.numpy(),
                "time": float(dx.time),
            }
        )
    return results


def _run_torch_vmap(m_mj, qvel_kick, nsteps, batch_size=4, disable_constraint=False):
    """Run mujoco-torch vmap simulation and return per-step results for env 0."""
    if disable_constraint:
        m_mj = m_mj.__copy__()
        m_mj.opt.disableflags = m_mj.opt.disableflags | int(DisableBit.CONSTRAINT)
    mx = mujoco_torch.device_put(m_mj)

    envs = []
    for _ in range(batch_size):
        d_mj = mujoco.MjData(m_mj)
        d_mj.qvel[:] = qvel_kick
        envs.append(mujoco_torch.device_put(d_mj))
    d_batch = torch.stack(envs, dim=0)

    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))

    results = []
    for _ in range(nsteps):
        d_batch = vmap_step(d_batch)
        results.append(
            {
                "qpos": d_batch.qpos[0].numpy(),
                "qvel": d_batch.qvel[0].numpy(),
                "act": d_batch.act[0].numpy(),
                "time": float(d_batch.time[0]),
            }
        )
    return results


def _compare_trajectories(mjx_results, torch_results, model_name, atol):
    """Compare two trajectories step by step."""
    for i, (mjx_step, torch_step) in enumerate(zip(mjx_results, torch_results)):
        for attr in ("qpos", "qvel", "act", "time"):
            mjx_val = np.asarray(mjx_step[attr])
            torch_val = np.asarray(torch_step[attr])
            np.testing.assert_allclose(
                torch_val,
                mjx_val,
                atol=atol,
                rtol=0,
                err_msg=f"{model_name}: {attr} mismatch at step {i + 1}",
            )


@pytest.mark.mjx
class TestMJXCorrectnessSingle:
    """Single-env mujoco-torch vs MJX."""

    def test_pendula(self):
        m_mj = test_util.load_test_file(SIMPLE_MODEL)
        rng = np.random.RandomState(SEED)
        qvel_kick = rng.randn(m_mj.nv) * 0.05

        mjx_results = _run_mjx(m_mj, qvel_kick, NSTEPS)
        torch_results = _run_torch_single(m_mj, qvel_kick, NSTEPS)
        _compare_trajectories(mjx_results, torch_results, SIMPLE_MODEL, atol=1e-8)

    def test_ant(self):
        m_mj = test_util.load_test_file(COMPLEX_MODEL)
        rng = np.random.RandomState(SEED)
        qvel_kick = rng.randn(m_mj.nv) * 0.05

        mjx_results = _run_mjx(m_mj, qvel_kick, NSTEPS)
        torch_results = _run_torch_single(m_mj, qvel_kick, NSTEPS)
        _compare_trajectories(mjx_results, torch_results, COMPLEX_MODEL, atol=1e-5)


@pytest.mark.mjx
class TestMJXCorrectnessVmap:
    """Batched (vmap) mujoco-torch vs MJX.

    Constraints are disabled because vmap + the solver's while_loop has
    known PyTorch limitations. The single-env tests above already verify
    full-fidelity constraint correctness. Only models without tendons
    are tested here (pendula.xml has tendons whose index_put_ is not
    vmap-compatible).
    """

    def test_ant_vmap(self):
        m_mj = test_util.load_test_file(COMPLEX_MODEL)
        rng = np.random.RandomState(SEED)
        qvel_kick = rng.randn(m_mj.nv) * 0.05

        mjx_results = _run_mjx(m_mj, qvel_kick, NSTEPS, disable_constraint=True)
        torch_results = _run_torch_vmap(m_mj, qvel_kick, NSTEPS, disable_constraint=True)
        _compare_trajectories(mjx_results, torch_results, COMPLEX_MODEL, atol=1e-5)

    def test_humanoid_vmap(self):
        m_mj = test_util.load_test_file("humanoid.xml")
        rng = np.random.RandomState(SEED)
        qvel_kick = rng.randn(m_mj.nv) * 0.05

        mjx_results = _run_mjx(m_mj, qvel_kick, NSTEPS, disable_constraint=True)
        torch_results = _run_torch_vmap(m_mj, qvel_kick, NSTEPS, disable_constraint=True)
        _compare_trajectories(mjx_results, torch_results, "humanoid.xml", atol=1e-5)
