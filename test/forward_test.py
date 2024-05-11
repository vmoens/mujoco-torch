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
import argparse

import pytest
from absl.testing import absltest
import mujoco
import mujoco_torch
from mujoco_torch._src import test_util
import numpy as np
import torch

# tolerance for difference between MuJoCo and MJX forward calculations - mostly
# due to float precision
_TOLERANCE = 1e-5


def _assert_eq(a, b, name, tol=_TOLERANCE):
  tol = tol * 10  # avoid test noise
  err_msg = f'mismatch: {name}:\n\tv0={a},\n\tv1={b}'
  torch.testing.assert_close(a, b, msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr, tol=_TOLERANCE):
  b = getattr(b, attr)
  a = torch.as_tensor(getattr(a, attr), dtype=b.dtype)
  _assert_eq(a, b, attr, tol=tol)


class TestForward:

  def test_forward(self):
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # apply some control and xfrc input
    d.ctrl = torch.tensor([-18, 0.59, 0.47])
    d.xfrc_applied[0, 2] = 0.1  # torque
    d.xfrc_applied[1, 4] = 0.3  # linear force
    mujoco.mj_step(m, d, 20)  # get some dynamics going
    mujoco.mj_forward(m, d)

    mx = mujoco_torch.put_model(m)

    # fwd_actuation
    # dx = torch.compile(mujoco_torch.fwd_actuation)(mx, mujoco_torch.put_data(m, d))
    dx = mujoco_torch.fwd_actuation(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'act_dot')
    _assert_attr_eq(d, dx, 'qfrc_actuator')

    # fwd_accleration (fwd_position and fwd_velocity already tested elsewhere)
    # dx = torch.compile(mujoco_torch.fwd_acceleration)(mx, mujoco_torch.put_data(m, d))
    dx = mujoco_torch.fwd_acceleration(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'qfrc_smooth')
    _assert_attr_eq(d, dx, 'qacc_smooth')

    # euler
    # dx = torch.compile(mujoco_torch.euler)(mx, mujoco_torch.put_data(m, d))
    dx = mujoco_torch.euler(mx, mujoco_torch.put_data(m, d))
    mujoco.mj_Euler(m, d)
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'qpos')
    _assert_attr_eq(d, dx, 'time')

  def test_step(self):
    m = test_util.load_test_file('constraints.xml')
    d = mujoco.MjData(m)
    # apply some control and xfrc input
    d.ctrl = torch.tensor([-18, 0.59, 0.47])
    d.xfrc_applied[0, 2] = 0.1  # torque
    d.xfrc_applied[1, 4] = 0.3  # linear force
    mujoco.mj_step(m, d, 20)  # get some dynamics going

    # dx = torch.compile(mujoco_torch.step)(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    dx = mujoco_torch.step(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    mujoco.mj_step(m, d)
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'time')
    _assert_attr_eq(d, dx, 'qvel', tol=5e-4)
    _assert_attr_eq(d, dx, 'qpos')

  def test_rk4(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option integrator="RK4">
            <flag constraint="disable"/>
          </option>
          <worldbody>
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

    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.array([0.2, -0.1])
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)

    # dx = torch.compile(mujoco_torch.rungekutta4)(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    dx = mujoco_torch.rungekutta4(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    mujoco.mj_RungeKutta(m, d, 4)

    _assert_attr_eq(d, dx, 'qvel')
    _assert_attr_eq(d, dx, 'qpos')
    _assert_attr_eq(d, dx, 'act')
    _assert_attr_eq(d, dx, 'time')

  def test_eulerdamp(self):
    m = test_util.load_test_file('pendula.xml')
    self.assertTrue((m.dof_damping > 0).any())

    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    mujoco.mj_forward(m, d)
    # dx = torch.compile(mujoco_torch.euler)(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    dx = mujoco_torch.euler(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    mujoco.mj_Euler(m, d)

    _assert_attr_eq(d, dx, 'qpos')

    # also test sparse
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    mujoco.mj_forward(m, d)
    # dx = torch.compile(mujoco_torch.euler)(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    dx = mujoco_torch.euler(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    mujoco.mj_Euler(m, d)

    _assert_attr_eq(d, dx, 'qpos')

  def test_disable_eulerdamp(self):
    m = test_util.load_test_file('pendula.xml')
    assert (m.dof_damping > 0).any()
    m.opt.disableflags = m.opt.disableflags | mujoco_torch.DisableBit.EULERDAMP

    d = mujoco.MjData(m)
    d.qvel[:] = 1.0
    d.qacc[:] = 1.0
    # dx = torch.compile(mujoco_torch.euler)(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))
    dx = mujoco_torch.euler(mujoco_torch.put_model(m), mujoco_torch.put_data(m, d))

    torch.testing.assert_close(dx.qvel, 1 + m.opt.timestep)


class ActuatorTest(absltest.TestCase):
  _DYN_XML = """
    <mujoco>
      <compiler autolimits="true"/>
      <worldbody>
        <body name="box">
          <joint name="slide1" type="slide" axis="1 0 0" />
          <joint name="slide2" type="slide" axis="0 1 0" />
          <joint name="slide3" type="slide" axis="0 0 1" />
          <joint name="slide4" type="slide" axis="1 1 0" />
          <geom type="box" size=".05 .05 .05" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <general joint="slide1" dynprm="0.1" gainprm="1.1" />
        <general joint="slide2" dyntype="integrator" dynprm="0.1" gainprm="1.1" />
        <general joint="slide3" dyntype="filter" dynprm="0.1" gainprm="1.1" />
        <general joint="slide4" dyntype="filterexact" dynprm="0.1" gainprm="1.1" />
      </actuator>
    </mujoco>
  """

  def test_dyntype(self):
    m = mujoco.MjModel.from_xml_string(self._DYN_XML)
    d = mujoco.MjData(m)
    d.ctrl = torch.tensor([1.5, 1.5, 1.5, 1.5])
    d.act = torch.tensor([0.5, 0.5, 0.5])

    mx = mujoco_torch.put_model(m)
    dx = mujoco_torch.put_data(m, d)

    mujoco.mj_fwdActuation(m, d)
    # dx = torch.compile(mujoco_torch.fwd_actuation)(mx, dx)
    dx = mujoco_torch.fwd_actuation(mx, dx)
    _assert_attr_eq(d, dx, 'act_dot')

    mujoco.mj_Euler(m, d)
    # dx = torch.compile(mujoco_torch.euler)(mx, dx)
    dx = mujoco_torch.euler(mx, dx)
    _assert_attr_eq(d, dx, 'act')


if __name__ == '__main__':
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
