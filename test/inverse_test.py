# Copyright 2025 DeepMind Technologies Limited
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
"""Tests for inverse dynamics functions."""

import mujoco
import numpy as np
from absl.testing import absltest, parameterized

import mujoco_torch
from mujoco_torch._src import test_util


def _assert_attr_eq(a, b, attr, fname, atol=1e-3, rtol=1e-3):
    err_msg = f"mismatch: {attr} in {fname}"
    a, b = getattr(a, attr), getattr(b, attr)
    np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class InverseTest(parameterized.TestCase):
    @parameterized.parameters(("pendula.xml",))
    def test_inverse_no_contact(self, fname):
        """Test inverse dynamics on a model without contacts."""
        np.random.seed(0)
        m = test_util.load_test_file(fname)
        d = mujoco.MjData(m)

        # kick and step forward to get non-trivial state and qacc
        d.qvel = np.random.random(m.nv) * 0.05
        for _ in range(10):
            mujoco.mj_step(m, d)

        # run MuJoCo C inverse as reference
        mujoco.mj_inverse(m, d)
        qfrc_inverse_ref = d.qfrc_inverse.copy()

        # run mujoco_torch inverse
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = mujoco_torch.inverse(mx, dx)

        np.testing.assert_allclose(
            dx.qfrc_inverse,
            qfrc_inverse_ref,
            atol=1e-8,
            rtol=1e-8,
            err_msg=f"qfrc_inverse mismatch in {fname}",
        )

    @parameterized.parameters(("ant.xml",), ("humanoid.xml",))
    def test_inverse_with_contact(self, fname):
        """Test inverse dynamics on models that generate contacts."""
        np.random.seed(test_util.TEST_FILES.index(fname))
        m = test_util.load_test_file(fname)
        d = mujoco.MjData(m)

        # step forward to generate contacts
        d.qvel = np.random.random(m.nv) * 0.05
        for _ in range(10):
            mujoco.mj_step(m, d)

        # run MuJoCo C inverse as reference
        mujoco.mj_inverse(m, d)
        qfrc_inverse_ref = d.qfrc_inverse.copy()

        # run mujoco_torch inverse
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = mujoco_torch.inverse(mx, dx)

        np.testing.assert_allclose(
            dx.qfrc_inverse,
            qfrc_inverse_ref,
            atol=1e-2,
            rtol=1e-2,
            err_msg=f"qfrc_inverse mismatch in {fname}",
        )

    def test_inverse_discrete_euler(self):
        """Test discrete inverse dynamics with Euler integrator."""
        np.random.seed(0)
        m = test_util.load_test_file("pendula.xml")
        m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        m.opt.enableflags = m.opt.enableflags | mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        d = mujoco.MjData(m)

        d.qvel = np.random.random(m.nv) * 0.05
        for _ in range(10):
            mujoco.mj_step(m, d)

        qacc_before = d.qacc.copy()

        mujoco.mj_inverse(m, d)
        qfrc_inverse_ref = d.qfrc_inverse.copy()

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = mujoco_torch.inverse(mx, dx)

        np.testing.assert_allclose(
            dx.qfrc_inverse,
            qfrc_inverse_ref,
            atol=1e-8,
            rtol=1e-8,
            err_msg="qfrc_inverse mismatch (discrete Euler)",
        )
        np.testing.assert_allclose(
            dx.qacc,
            qacc_before,
            atol=1e-12,
            err_msg="qacc not restored after discrete inverse",
        )

    def test_inverse_discrete_implicitfast(self):
        """Test discrete inverse dynamics with ImplicitFast integrator."""
        np.random.seed(1)
        # Use a model without actuators to avoid the deriv_smooth_vel act bug
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option integrator="implicitfast">
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <body pos="0.15 0 0">
              <joint type="hinge" axis="0 1 0" damping="5"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
              <body pos="0.1 0 0">
                <joint type="hinge" axis="0 1 0" damping="3"/>
                <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)
        m.opt.enableflags = m.opt.enableflags | mujoco.mjtEnableBit.mjENBL_INVDISCRETE
        d = mujoco.MjData(m)

        d.qvel = np.random.random(m.nv) * 0.05
        for _ in range(10):
            mujoco.mj_step(m, d)

        qacc_before = d.qacc.copy()

        mujoco.mj_inverse(m, d)
        qfrc_inverse_ref = d.qfrc_inverse.copy()

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = mujoco_torch.inverse(mx, dx)

        np.testing.assert_allclose(
            dx.qfrc_inverse,
            qfrc_inverse_ref,
            atol=1e-8,
            rtol=1e-8,
            err_msg="qfrc_inverse mismatch (discrete ImplicitFast)",
        )
        np.testing.assert_allclose(
            dx.qacc,
            qacc_before,
            atol=1e-12,
            err_msg="qacc not restored after discrete inverse",
        )


if __name__ == "__main__":
    absltest.main()
