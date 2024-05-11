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
"""Tests for moving mujoco structs on and off device."""
import argparse
import torch
import pytest
import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import mujoco
from tensordict import is_tensorclass

import mujoco_torch
from mujoco_torch._src import device
from mujoco_torch._src import test_util
from mujoco_torch._src import types
# pylint: disable=g-importing-member
from mujoco_torch._src.dataclasses import PyTreeNode
# pylint: enable=g-importing-member
import numpy as np


def _assert_eq(a, b, attr=None, name=None):
  if (type(a), attr) in device._DERIVED:
    return


  if is_tensorclass(a):
    for key, val in a.items():
      _assert_eq(val, getattr(b, key), attr=key, name=a.__class__.__name__)
      return

  typ = {'Model': types.Model, 'Data': types.Data,
         'Contact': types.Contact}.get(name)
  if (typ, attr) in device._TRANSFORMS:
    b = device._TRANSFORMS[(typ, attr)](b)

  # if not hasattr(b, 'shape') or not b.shape:
  #   assert a == b, err_msg
  #   return
  if not isinstance(b, torch.Tensor):
    b = torch.as_tensor(b, dtype=a.dtype)

  a, b = torch.broadcast_tensors(a, b)
  err_msg = f'mismatch: {attr} in {name} (a.shape={a.shape} - b.shape={b.shape}, \na={a}, \nb={b})'
  torch.testing.assert_close(a, b, msg=err_msg)


class TestDevice:

  @pytest.mark.parametrize("fname",['constraints.xml', 'pendula.xml'])
  def testdevice_put(self, fname):
    """Test putting MjData and MjModel on device."""
    m = test_util.load_test_file(fname)
    # advance state to ensure non-zero fields
    d = mujoco.MjData(m)
    for _ in range(10):
      mujoco.mj_step(m, d)

    _assert_eq(mujoco_torch.device_put(d), d)
    _assert_eq(mujoco_torch.device_put(m), m)

  @pytest.mark.parametrize("fname",['constraints.xml', 'pendula.xml'])
  def testdevice_get(self, fname):
    """Test getting MjData from a device."""
    m = test_util.load_test_file(fname)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE  # force sparse for testing
    mx = device.device_put(m)
    dx = mujoco_torch.make_data(mx)
    d = mujoco.MjData(m)
    print('dx.solver_niter', dx.solver_niter)
    print('d.solver_niter', d.solver_niter)
    device.device_get_into(d, dx)
    print('dx.solver_niter', dx.solver_niter)
    print('d.solver_niter', d.solver_niter)
    _assert_eq(dx, d)

  @pytest.mark.parametrize("fname",['constraints.xml', 'pendula.xml'])
  def testdevice_get_batched(self, fname):
    """Test getting MjData from a device."""
    m = test_util.load_test_file(fname)
    m.opt.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE  # force sparse for testing
    mx = device.device_put(m)
    batch_size = 32

    # create mjx_data and batch it
    dx = mujoco_torch.make_data(mx)
    dx = torch.utils._pytree.tree_map(
        lambda x: x.expand(batch_size, *x.shape),
        dx,
    )
    ds = [mujoco.MjData(m) for _ in range(batch_size - 1)]

    with pytest.raises(ValueError):
      device.device_get_into(ds, dx)

    ds = [mujoco.MjData(m) for _ in range(batch_size)]
    device.device_get_into(ds, dx)
    dx = dx.cpu()  # faster indexing for testing
    for i in range(batch_size):
      _assert_eq(torch.utils._pytree.tree_map(lambda x, i=i: x[i], dx), ds[i])


class TestValidateInput:

  def test_solver(self):
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option solver="PGS"/><worldbody/></mujoco>'
    )
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_integrator(self):
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option integrator="implicit"/><worldbody/></mujoco>'
    )
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_cone(self):
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><option cone="elliptic"/><worldbody/></mujoco>'
    )
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_dyn(self):
    m = test_util.load_test_file('pendula.xml')
    m.actuator_dyntype[0] = mujoco.mjtDyn.mjDYN_MUSCLE
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_gain(self):
    m = test_util.load_test_file('pendula.xml')
    m.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_MUSCLE
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_bias(self):
    m = test_util.load_test_file('pendula.xml')
    m.actuator_gaintype[0] = mujoco.mjtGain.mjGAIN_MUSCLE
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_condim(self):
    m = test_util.load_test_file('constraints.xml')
    for i in [1, 4, 6]:
      m.geom_condim[0] = i
      with pytest.raises(NotImplementedError):
        mujoco_torch.device_put(m)

  def test_geoms(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
          <body>
            <joint axis="1 0 0" type="free"/>
            <geom size="0.2 0.2 0.2" type="box"/>
          </body>
          <body>
            <joint axis="1 0 0" type="free"/>
            <geom size="0.1 0.1" type="cylinder"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)

  def test_tendon(self):
    m = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <worldbody>
        <body name="left_thigh" pos="0 0.1 -0.04">
          <joint axis="0 1 0" name="left_hip_y" type="hinge"/>
          <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>
          <body name="left_shin" pos="0 -0.01 -0.403">
            <joint axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>
          </body>
        </body>
        </worldbody>
        <tendon>
          <fixed name="left_hipknee">
            <joint coef="-1" joint="left_hip_y"/>
            <joint coef="1" joint="left_knee"/>
          </fixed>
        </tendon>
      </mujoco>
    """)
    with pytest.raises(NotImplementedError):
      mujoco_torch.device_put(m)


if __name__ == '__main__':
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
