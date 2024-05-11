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
"""Tests for smooth dynamics functions."""

import argparse
import pytest
import torch
from absl.testing import absltest
import jax
import mujoco
import mujoco_torch
from mujoco_torch._src import test_util
import numpy as np

# tolerance for difference between MuJoCo and MJX smooth calculations - mostly
# due to float precision
_TOLERANCE = 5e-5


def _assert_eq(a, b, name):
  tol = _TOLERANCE * 10  # avoid test noise
  err_msg = f'mismatch: {name}'
  torch.testing.assert_allclose(a, b, msg=err_msg, atol=tol, rtol=tol)


def _assert_attr_eq(a, b, attr):
  _assert_eq(getattr(a, attr), getattr(b, attr), attr)


class TestSmooth:

  def setUp(self):
    super().setUp()
    # although we already have generous padding of thresholds, it doesn't hurt
    # to also fix the seed to reduce test flakiness
    np.random.seed(0)

  def test_smooth(self):
    """Tests MJX smooth functions match MuJoCo smooth functions."""

    m = test_util.load_test_file('pendula.xml')
    # # force MJX sparse for testing:
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    d = mujoco.MjData(m)
    # give the system a little kick to ensure we have non-identity rotations
    d.qvel = np.random.random(m.nv)
    mujoco.mj_step(m, d, 10)  # let dynamics get state significantly non-zero
    mujoco.mj_forward(m, d)
    mx = mujoco_torch.put_model(m)

    # kinematics
    # dx = torch.compile(mujoco_torch.kinematics)(mx, mujoco_torch.put_data(m, d))
    dx = mujoco_torch.kinematics(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'xanchor')
    _assert_attr_eq(d, dx, 'xaxis')
    _assert_attr_eq(d, dx, 'xpos')
    _assert_attr_eq(d, dx, 'xquat')
    _assert_eq(d.xmat.reshape((-1, 3, 3)), dx.xmat, 'xmat')
    _assert_attr_eq(d, dx, 'xipos')
    _assert_eq(d.ximat.reshape((-1, 3, 3)), dx.ximat, 'ximat')
    _assert_attr_eq(d, dx, 'geom_xpos')
    _assert_eq(d.geom_xmat.reshape((-1, 3, 3)), dx.geom_xmat, 'geom_xmat')
    _assert_attr_eq(d, dx, 'site_xpos')
    _assert_eq(d.site_xmat.reshape((-1, 3, 3)), dx.site_xmat, 'site_xmat')
    # com_pos
    dx = torch.compile(mujoco_torch.com_pos)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'subtree_com')
    _assert_attr_eq(d, dx, 'cinert')
    _assert_attr_eq(d, dx, 'cdof')
    # crb
    dx = torch.compile(mujoco_torch.crb)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'crb')
    _assert_attr_eq(d, dx, 'qM')
    # factor_m
    dx = torch.compile(mujoco_torch.factor_m)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'qLD')
    _assert_attr_eq(d, dx, 'qLDiagInv')
    # com_vel
    dx = torch.compile(mujoco_torch.com_vel)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'cvel')
    _assert_attr_eq(d, dx, 'cdof_dot')
    # rne
    dx = torch.compile(mujoco_torch.rne)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'qfrc_bias')
    # transmission
    dx = torch.compile(mujoco_torch.transmission)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'actuator_length')
    _assert_attr_eq(d, dx, 'actuator_moment')
    # camlight
    dx = torch.compile(mujoco_torch.camlight)(mx, mujoco_torch.put_data(m, d))
    _assert_attr_eq(d, dx, 'cam_xpos')
    _assert_eq(d.cam_xmat.reshape((-1, 3, 3)), dx.cam_xmat, 'cam_xmat')

  def test_disable_gravity(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option>
            <flag gravity="disable"/>
          </option>
          <worldbody>
            <body>
              <joint type="free"/>
              <geom size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mujoco_torch.put_model(m)
    dx = mujoco_torch.put_data(m, d)

    dx = torch.compile(mujoco_torch.rne)(mx, dx)
    np.testing.assert_allclose(dx.qfrc_bias, 0)

  def test_site_transmission(self):
    m = mujoco.MjModel.from_xml_string("""
        <mujoco>
        <compiler autolimits="true"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
            <site name="site1"/>
            <body>
              <joint type="hinge"/>
              <geom size="0.1" mass="1"/>
              <site name="site2" pos="0.1 0.2 0.3"/>
            </body>
          </body>
          <body pos="1 0 0">
            <joint name="slide" type="hinge"/>
            <geom type="box" size=".05 .05 .05" mass="1"/>
          </body>
        </worldbody>
        <actuator>
          <position site="site1" kv="0.1" gear="1 2 3 0 0 0"/>
          <position site="site1" kv="0.2" gear="0 0 0 1 2 3"/>
          <position site="site2" kv="0.3" gear="0 3 0 0 0 1"/>
          <position joint="slide" kv="0.05" />
          <position site="site2" refsite="site1" gear="1 2 3 0.5 0.4 0.6"/>
        </actuator>
        </mujoco>
      """)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mx = mujoco_torch.put_model(m)
    dx = mujoco_torch.put_data(m, d)

    mujoco.mj_transmission(m, d)
    dx = torch.compile(mujoco_torch.transmission)(mx, dx)
    _assert_attr_eq(d, dx, 'actuator_length')
    _assert_attr_eq(d, dx, 'actuator_moment')



if __name__ == '__main__':
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
