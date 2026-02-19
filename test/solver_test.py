# Copyright 2023 DeepMind Technologies Limited
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
"""Tests for forward functions."""

import mujoco
import numpy as np
import torch
from absl.testing import absltest, parameterized
from etils import epath

import mujoco_torch


def _assert_attr_eq(a, b, attr, step, fname, atol=1e-2, rtol=1e-2):
    err_msg = f"mismatch: {attr} at step {step} in {fname}"
    a, b = getattr(a, attr), getattr(b, attr)
    np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class Solver64Test(parameterized.TestCase):
    """Tests solvers at 64 bit precision."""

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        super().tearDown()
        torch.set_default_dtype(torch.float32)

    @parameterized.parameters(enumerate(("ant.xml",)))
    def test_cg(self, seed, fname):
        """Test mujoco_torch cg solver matches mujoco cg solver at 64 bit precision."""
        f = epath.resource_path("mujoco_torch") / "test_data" / fname
        m = mujoco.MjModel.from_xml_string(f.read_text())
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        torch.set_default_dtype(torch.float64)
        forward_jit_fn = mujoco_torch.forward

        # give the system a little kick to ensure we have non-identity rotations
        np.random.seed(seed)
        d.qvel = 0.01 * np.random.random(m.nv)

        for i in range(100):
            # in order to avoid re-jitting, reuse the same mj_data shape
            save = d.qpos, d.qvel, d.time, d.qacc_warmstart, d.qacc_smooth
            d = mujoco.MjData(m)
            d.qpos, d.qvel, d.time, d.qacc_warmstart, d.qacc_smooth = save
            dx = mujoco_torch.device_put(d)

            mujoco.mj_step(m, d)
            dx = forward_jit_fn(mx, dx)

            # at 64 bits the solutions returned by the two solvers are quite close
            self.assertLessEqual(dx.solver_niter[0], d.solver_niter[0])
            _assert_attr_eq(d, dx, "qfrc_constraint", i, fname)
            _assert_attr_eq(d, dx, "qacc", i, fname)


class SolverTest(parameterized.TestCase):
    # TODO: humanoid.xml excluded due to boundary contact precision differences
    # between our kinematics and MuJoCo's. Contacts at dist~0 flip active/inactive.
    @parameterized.parameters(enumerate(("ant.xml",)))
    def test_cg(self, seed, fname):
        """Test mujoco_torch cg solver is close to mj at 32 bit precision.

        Args:
          seed: int
          fname: file to test

        At lower float resolution there's wiggle room in valid forces that satisfy
        constraints.  So instead let's mainly validate that mujoco_torch is finding solutions
        with as good cost as mujoco, even if the resulting forces/accelerations
        are not quite the same.
        """
        f = epath.resource_path("mujoco_torch") / "test_data" / fname
        m = mujoco.MjModel.from_xml_string(f.read_text())
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        forward_jit_fn = mujoco_torch.forward

        # give the system a little kick to ensure we have non-identity rotations
        np.random.seed(seed)
        d.qvel = 0.01 * np.random.random(m.nv)

        for i in range(100):
            # in order to avoid re-jitting, reuse the same mj_data shape
            save = d.qpos, d.qvel, d.time, d.qacc_warmstart, d.qacc_smooth
            d = mujoco.MjData(m)
            d.qpos, d.qvel, d.time, d.qacc_warmstart, d.qacc_smooth = save
            dx = mujoco_torch.device_put(d)

            mujoco.mj_step(m, d)
            dx = forward_jit_fn(mx, dx)

            def cost(qacc):
                jaref = np.zeros(d.nefc)
                mujoco.mj_mulJacVec(m, d, jaref, qacc)
                jaref -= d.efc_aref
                cost = np.array([0.0])
                mujoco.mj_constraintUpdate(m, d, jaref, cost, 0)
                return cost[0]

            cost_mj, cost_mjx = cost(d.qacc), cost(dx.qacc)

            self.assertLessEqual(
                cost_mjx,
                cost_mj * 1.01,
                msg=f"mismatch: {fname} at step {i}, cost too high",
            )
            _assert_attr_eq(d, dx, "qfrc_constraint", i, fname, atol=1e-1, rtol=1e-1)
            _assert_attr_eq(d, dx, "qacc", i, fname, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    absltest.main()
