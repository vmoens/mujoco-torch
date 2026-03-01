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

# pylint: enable=g-importing-member
import numpy as np

# from torch import numpy as torch
import torch
from absl.testing import absltest, parameterized

import mujoco_torch
from mujoco_torch._src import forward, test_util

# pylint: disable=g-importing-member
from mujoco_torch._src.types import DisableBit


def _assert_attr_eq(a, b, attr, step, fname, atol=1e-3, rtol=1e-3):
    err_msg = f"mismatch: {attr} at step {step} in {fname}"
    a, b = getattr(a, attr), getattr(b, attr)
    np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class ForwardTest(parameterized.TestCase):
    @parameterized.parameters(filter(lambda s: s not in ("equality.xml",), test_util.TEST_FILES))
    def test_forward(self, fname):
        """Test mujoco mj forward function matches mujoco_mjx forward function."""
        np.random.seed(test_util.TEST_FILES.index(fname))

        m = test_util.load_test_file(fname)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.make_data(mx)
        forward_jit_fn = mujoco_torch.forward

        # give the system a little kick to ensure we have non-identity rotations
        d.qvel = np.random.random(m.nv) * 0.05
        for i in range(100):
            qpos, qvel = torch.tensor(d.qpos.copy()), torch.tensor(d.qvel.copy())
            mujoco.mj_step(m, d)
            dx = forward_jit_fn(mx, dx.replace(qpos=qpos, qvel=qvel))

            _assert_attr_eq(d, dx, "qfrc_smooth", i, fname)
            _assert_attr_eq(d, dx, "qacc_smooth", i, fname)

    @parameterized.parameters(filter(lambda s: s not in ("equality.xml",), test_util.TEST_FILES))
    def test_step(self, fname):
        """Test mujoco mj step matches mujoco_mjx step."""
        np.random.seed(test_util.TEST_FILES.index(fname))
        m = test_util.load_test_file(fname)
        step_jit_fn = forward.step

        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        # give the system a little kick to ensure we have non-identity rotations
        d.qvel = np.random.normal(m.nv) * 0.05
        for i in range(100):
            # in order to avoid re-jitting, reuse the same mj_data shape
            qpos, qvel = d.qpos, d.qvel
            d = mujoco.MjData(m)
            d.qpos, d.qvel = qpos, qvel
            dx = mujoco_torch.device_put(d)

            mujoco.mj_step(m, d)
            dx = step_jit_fn(mx, dx)

            _assert_attr_eq(d, dx, "qvel", i, fname, atol=1e-2)
            _assert_attr_eq(d, dx, "qpos", i, fname, atol=1e-2)
            _assert_attr_eq(d, dx, "act", i, fname)
            _assert_attr_eq(d, dx, "time", i, fname)

    def test_rk4(self):
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option integrator="RK4">
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <light pos="0 0 1"/>
            <geom type="plane" size="1 1 .01" pos="0 0 -1"/>
            <body pos="0.15 0 0">
              <joint type="hinge" axis="0 1 0"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
              <body pos="0.1 0 0">
                <joint type="slide" axis="1 0 0" stiffness="200"/>
                <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """)
        step_jit_fn = forward.step

        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        # give the system a little kick to ensure we have non-identity rotations
        d.qvel = np.random.normal(m.nv) * 0.05
        for i in range(100):
            # in order to avoid re-jitting, reuse the same mj_data shape
            qpos, qvel = d.qpos, d.qvel
            d = mujoco.MjData(m)
            d.qpos, d.qvel = qpos, qvel
            dx = mujoco_torch.device_put(d)

            mujoco.mj_step(m, d)
            dx = step_jit_fn(mx, dx)

            _assert_attr_eq(d, dx, "qvel", i, "test_rk4", atol=1e-2)
            _assert_attr_eq(d, dx, "qpos", i, "test_rk4", atol=1e-2)
            _assert_attr_eq(d, dx, "act", i, "test_rk4")
            _assert_attr_eq(d, dx, "time", i, "test_rk4")

    def test_disable_eulerdamp(self):
        m = test_util.load_test_file("ant.xml")
        m.opt.disableflags = m.opt.disableflags | DisableBit.EULERDAMP

        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        self.assertTrue((mx.dof_damping > 0).any())
        dx = mujoco_torch.device_put(d)
        dx = forward.forward(mx, dx)

        dx = dx.replace(qvel=torch.ones_like(dx.qvel), qacc=torch.ones_like(dx.qacc))
        dx = forward._euler(mx, dx)
        np.testing.assert_allclose(dx.qvel, 1 + m.opt.timestep)

    @parameterized.parameters("ant.xml", "humanoid.xml")
    def test_step_vmap(self, fname):
        """Test that vmap(step) matches sequential step for each env in a batch.

        Constraints are disabled so the test isolates the forward/integration
        pipeline (including transmission) without hitting the solver's
        while_loop, which has its own vmap limitations.
        """
        torch.set_default_dtype(torch.float64)
        batch_size = 4
        nsteps = 5

        m = test_util.load_test_file(fname)
        m.opt.disableflags = m.opt.disableflags | DisableBit.CONSTRAINT
        mx = mujoco_torch.device_put(m)

        rng = np.random.RandomState(42)
        envs = []
        for _ in range(batch_size):
            d = mujoco.MjData(m)
            d.qvel[:] = 0.01 * rng.randn(m.nv)
            envs.append(mujoco_torch.device_put(d))

        # sequential reference
        seq_results = []
        for dx in envs:
            for _ in range(nsteps):
                dx = mujoco_torch.step(mx, dx)
            seq_results.append(dx)

        # batched via vmap
        d_batch = torch.stack(envs, dim=0)
        vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
        for _ in range(nsteps):
            d_batch = vmap_step(d_batch)

        for i in range(batch_size):
            for attr in ("qpos", "qvel"):
                np.testing.assert_allclose(
                    getattr(d_batch[i], attr).detach().numpy(),
                    getattr(seq_results[i], attr).detach().numpy(),
                    atol=1e-8,
                    err_msg=f"vmap vs sequential mismatch: {attr} env={i} in {fname}",
                )

    def test_filterexact(self):
        """Test FILTEREXACT actuator dynamics match MuJoCo C."""
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option>
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <body>
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <general joint="slide" dyntype="filterexact" dynprm="0.05"
                     gainprm="100" biastype="affine" biasprm="0 -100 0"/>
          </actuator>
        </mujoco>
        """)
        d = mujoco.MjData(m)
        d.ctrl[0] = 1.0

        mx = mujoco_torch.device_put(m)

        for i in range(200):
            qpos, qvel, act = d.qpos.copy(), d.qvel.copy(), d.act.copy()
            d_new = mujoco.MjData(m)
            d_new.qpos[:] = qpos
            d_new.qvel[:] = qvel
            d_new.act[:] = act
            d_new.ctrl[0] = 1.0
            dx = mujoco_torch.device_put(d_new)

            mujoco.mj_step(m, d)
            dx = forward.step(mx, dx)

            np.testing.assert_allclose(
                dx.act.numpy(),
                d.act,
                atol=1e-8,
                err_msg=f"act mismatch at step {i}",
            )
            np.testing.assert_allclose(
                dx.qpos.numpy(),
                d.qpos,
                atol=1e-5,
                err_msg=f"qpos mismatch at step {i}",
            )

    def test_muscle(self):
        """Test MUSCLE actuator dynamics, gain, and bias match MuJoCo C."""
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <compiler autolimits="true"/>
          <option>
            <flag constraint="disable"/>
          </option>
          <worldbody>
            <body>
              <joint name="hinge" type="hinge" axis="0 0 1"
                     range="-90 90" limited="true"/>
              <geom type="capsule" size="0.05" fromto="0 0 0 0.3 0 0" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <muscle joint="hinge" lengthrange="0.5 1.5"
                    timeconst="0.01 0.04"/>
          </actuator>
        </mujoco>
        """)
        d = mujoco.MjData(m)
        d.ctrl[0] = 0.8

        mx = mujoco_torch.device_put(m)

        for i in range(200):
            qpos, qvel, act = d.qpos.copy(), d.qvel.copy(), d.act.copy()
            d_new = mujoco.MjData(m)
            d_new.qpos[:] = qpos
            d_new.qvel[:] = qvel
            d_new.act[:] = act
            d_new.ctrl[0] = 0.8
            dx = mujoco_torch.device_put(d_new)

            mujoco.mj_step(m, d)
            dx = forward.step(mx, dx)

            np.testing.assert_allclose(
                dx.act.numpy(),
                d.act,
                atol=1e-8,
                err_msg=f"act mismatch at step {i}",
            )
            np.testing.assert_allclose(
                dx.qpos.numpy(),
                d.qpos,
                atol=1e-4,
                err_msg=f"qpos mismatch at step {i}",
            )


if __name__ == "__main__":
    absltest.main()
