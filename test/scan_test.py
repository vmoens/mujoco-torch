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
"""Tests for scan functions."""

import mujoco

# pylint: enable=g-importing-member
import numpy as np

# from jax import numpy as torch
import torch
from absl.testing import absltest

import mujoco_torch

# pylint: disable=g-importing-member
from mujoco_torch._src import scan


class ScanTest(absltest.TestCase):
    _MULTI_DOF_XML = """
      <mujoco>
        <compiler inertiafromgeom="true"/>
        <worldbody>
          <body>
            <joint type="free"/>
            <geom size=".15" mass="1" type="sphere"/>
            <body>
              <joint axis="1 0 0" pos="1 0 0" type="ball"/>
              <geom size=".15" mass="2" type="sphere"/>
            </body>
            <body>
              <joint axis="1 0 0" pos="2 0 0" type="hinge"/>
              <joint axis="0 1 0" pos="3 0 0" type="slide"/>
              <geom size=".15" mass="3" type="sphere"/>
            </body>
          </body>
        </worldbody>
      </mujoco>
    """

    def test_flat_empty(self):
        """Test scanning over just world body."""
        m = mujoco.MjModel.from_xml_string("""
      <mujoco model="world_body">
        <worldbody/>
      </mujoco>
    """)
        m = mujoco_torch.device_put(m)

        def fn(body_id):
            return body_id + 1

        b_in = torch.tensor([1])
        b_expect = torch.tensor([2])
        b_out = scan.flat(m, fn, "b", "b", b_in)

        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

    def test_flat_joints(self):
        """Tests scanning over bodies with joints of different types."""
        m = mujoco.MjModel.from_xml_string(self._MULTI_DOF_XML)
        m = mujoco_torch.device_put(m)

        # we will test two functions:
        #   1) j_fn receives jnt_types as a torch array
        #   2) s_fn receives jnt_types as a static array and can switch on it
        j_fn = lambda jnt_pos, val: val + torch.sum(jnt_pos)
        s_fn = lambda jnt_types, val: val + sum(jnt_types)

        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        b_expect = torch.tensor([[0, 0], [1, 1], [3, 3], [8, 8]])
        b_out = scan.flat(m, j_fn, "jb", "b", m.jnt_pos, b_in)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

        b_out = scan.flat(m, s_fn, "jb", "b", m.jnt_type, b_in)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

        # we should not call functions for which we know we will discard the results
        def no_world(jnt_types, val):
            if len(jnt_types) == 0:
                self.fail("world has no dofs, should not be called")
            return val + sum(jnt_types)

        v_in = torch.ones((m.nv, 1))
        scan.flat(m, no_world, "jv", "v", m.jnt_type, v_in)

    def test_body_tree(self):
        """Tests tree scanning over bodies with different joint counts."""
        m = mujoco.MjModel.from_xml_string(self._MULTI_DOF_XML)
        m = mujoco_torch.device_put(m)

        # we will test two functions:
        #   1) j_fn receives jnt_pos which is a torch array
        #   2) s_fn receives jnt_types which is a static array
        def j_fn(carry, jnt_pos, val):
            carry = torch.zeros_like(val) if carry is None else carry
            return carry + val + torch.sum(jnt_pos)

        def s_fn(carry, jnt_types, val):
            carry = torch.zeros_like(val) if carry is None else carry
            return carry + val + sum(jnt_types)

        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])
        b_expect = torch.tensor([[0, 0], [1, 1], [4, 4], [9, 9]])

        b_out = scan.body_tree(m, j_fn, "jb", "b", m.jnt_pos, b_in)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

        b_out = scan.body_tree(m, s_fn, "jb", "b", m.jnt_type, b_in)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

        # reverse:
        b_expect = torch.tensor([[12, 12], [12, 12], [3, 3], [8, 8]])
        b_out = scan.body_tree(m, j_fn, "jb", "b", m.jnt_pos, b_in, reverse=True)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

        b_out = scan.body_tree(m, s_fn, "jb", "b", m.jnt_type, b_in, reverse=True)
        np.testing.assert_equal(np.array(b_out), np.array(b_expect))

    _MULTI_ACT_XML = """
    <mujoco>
      <option timestep="0.02"/>
      <compiler autolimits="true"/>
      <default>
        <geom contype="0" conaffinity="0"/>
      </default>
      <worldbody>
        <body pos="0 0 -0.5">
          <joint type="free" name="joint0" range="-37.81 86.15"/>
          <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
          <body pos="0 0 -0.5">
            <joint axis="1 0 0" type="hinge" name="joint1" range="-32.88 5.15" actuatorfrcrange="-0.48 0.72"/>
            <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
            <body pos="0 0 -0.5">
              <joint axis="0 1 0" type="slide" name="joint2" range="-83.31 54.77"/>
              <joint axis="1 0 0" type="slide" name="joint3" range="-87.84 60.58" actuatorfrcrange="-0.62 0.28"/>
              <joint axis="0 1 0" type="hinge" name="joint4" range="-49.96 83.03" actuatorfrcrange="-0.01 0.64"/>
              <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
              <body pos="0 0 -0.5">
                <joint axis="1 0 0" type="slide" name="joint5" range="-1.53 87.07" actuatorfrcrange="-0.06 0.26"/>
                <joint axis="0 1 0" type="slide" name="joint6" range="-63.57 9.95"/>
                <joint axis="1 0 0" type="hinge" name="joint7" range="-22.50 41.20" actuatorfrcrange="-0.41 0.84"/>
                <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
                <body pos="0 0 -0.5">
                  <joint axis="0 1 0" type="hinge" name="joint8" range="-34.67 72.56"/>
                  <joint axis="1 0 0" type="hinge" name="joint9" range="-1.81 16.22" actuatorfrcrange="-0.41 0.14"/>
                  <joint axis="1 0 0" type="ball" name="joint10" range="0.00 6.02" actuatorfrcrange="-0.74 0.12"/>
                  <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position joint="joint6" gear="1"/>
        <intvelocity joint="joint1" kp="2000" actrange="-0.7 2.3"/>
        <motor joint="joint3" gear="42"/>
        <velocity joint="joint2" kv="123"/>
        <position joint="joint0" ctrlrange="-0.9472 0.9472"/>
        <intvelocity joint="joint7" kp="2000" actrange="-0.7 2.3"/>
        <general joint="joint5" ctrlrange="0.1 2.34346" biastype="affine" gainprm="35 0 0" biasprm="0 -35 -0.65"/>
        <general joint="joint4" ctrlrange="0.1 2.34346" biastype="affine" gainprm="35 0 0" biasprm="0 -35 -0.65"/>
      </actuator>
    </mujoco>
  """

    def testscan_actuators(self):
        """Tests scanning over actuators."""
        m = mujoco.MjModel.from_xml_string(self._MULTI_ACT_XML)
        m = mujoco_torch.device_put(m)

        fn = lambda *args: args
        args = (
            m.actuator_gear,
            m.jnt_type,
            torch.arange(m.nq),
            torch.arange(m.nv),
            torch.tensor([1.4, 1.1]),
        )
        gear, jnt_typ, qadr, vadr, act = scan.flat(m, fn, "ujqva", "ujqva", *args, group_by="u")

        np.testing.assert_array_equal(gear, m.actuator_gear)
        np.testing.assert_array_equal(jnt_typ, m.jnt_type.data[m.actuator_trnid[:, 0]])
        np.testing.assert_array_equal(act, torch.tensor([1.4, 1.1]))
        expected_vadr = np.concatenate([np.nonzero(m.dof_jntid == trnid)[0] for trnid in m.actuator_trnid[:, 0]])
        np.testing.assert_array_equal(vadr, expected_vadr)
        expected_qadr = np.concatenate([np.nonzero(scan._q_jointid(m) == i)[0] for i in m.actuator_trnid[:, 0]])
        np.testing.assert_array_equal(qadr, expected_qadr)


class ScanPaddingTest(absltest.TestCase):
    """Tests verifying numerical equivalence of padded and non-padded paths."""

    _MULTI_DOF_XML = ScanTest._MULTI_DOF_XML
    _MULTI_ACT_XML = ScanTest._MULTI_ACT_XML

    def _models(self, xml):
        m_mj = mujoco.MjModel.from_xml_string(xml)
        m_no_pad = mujoco_torch.device_put(m_mj)
        m_pad = mujoco_torch.device_put(m_mj, scan_padding=True)
        return m_no_pad, m_pad

    def test_flat_padding_equivalence(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        j_fn = lambda jnt_pos, val: val + torch.sum(jnt_pos)
        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        out_np = scan.flat(m_np, j_fn, "jb", "b", m_np.jnt_pos, b_in)
        out_p = scan.flat(m_p, j_fn, "jb", "b", m_p.jnt_pos, b_in)
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

    def test_flat_padding_static_np(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        s_fn = lambda jnt_types, val: val + sum(jnt_types)
        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        out_np = scan.flat(m_np, s_fn, "jb", "b", m_np.jnt_type, b_in)
        out_p = scan.flat(m_p, s_fn, "jb", "b", m_p.jnt_type, b_in)
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

    def test_flat_padding_v_output(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        def fn(jnt_types, v_in):
            return v_in * 2

        v_in = torch.ones((m_np.nv, 1))
        out_np = scan.flat(m_np, fn, "jv", "v", m_np.jnt_type, v_in)
        out_p = scan.flat(m_p, fn, "jv", "v", m_p.jnt_type, v_in)
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

    def test_flat_padding_actuators(self):
        m_np, m_p = self._models(self._MULTI_ACT_XML)

        fn = lambda *args: args
        args_np = (
            m_np.actuator_gear,
            m_np.jnt_type,
            torch.arange(m_np.nq),
            torch.arange(m_np.nv),
            torch.tensor([1.4, 1.1]),
        )
        args_p = (
            m_p.actuator_gear,
            m_p.jnt_type,
            torch.arange(m_p.nq),
            torch.arange(m_p.nv),
            torch.tensor([1.4, 1.1]),
        )
        out_np = scan.flat(
            m_np,
            fn,
            "ujqva",
            "ujqva",
            *args_np,
            group_by="u",
        )
        out_p = scan.flat(
            m_p,
            fn,
            "ujqva",
            "ujqva",
            *args_p,
            group_by="u",
        )
        for a, b in zip(out_np, out_p):
            np.testing.assert_array_equal(np.array(a), np.array(b))

    def test_body_tree_padding_forward(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        def j_fn(carry, jnt_pos, val):
            carry = torch.zeros_like(val) if carry is None else carry
            return carry + val + torch.sum(jnt_pos)

        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        out_np = scan.body_tree(
            m_np,
            j_fn,
            "jb",
            "b",
            m_np.jnt_pos,
            b_in,
        )
        out_p = scan.body_tree(
            m_p,
            j_fn,
            "jb",
            "b",
            m_p.jnt_pos,
            b_in,
        )
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

    def test_body_tree_padding_reverse(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        def j_fn(carry, jnt_pos, val):
            carry = torch.zeros_like(val) if carry is None else carry
            return carry + val + torch.sum(jnt_pos)

        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        out_np = scan.body_tree(
            m_np,
            j_fn,
            "jb",
            "b",
            m_np.jnt_pos,
            b_in,
            reverse=True,
        )
        out_p = scan.body_tree(
            m_p,
            j_fn,
            "jb",
            "b",
            m_p.jnt_pos,
            b_in,
            reverse=True,
        )
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

    def test_body_tree_padding_static_np(self):
        m_np, m_p = self._models(self._MULTI_DOF_XML)

        def s_fn(carry, jnt_types, val):
            carry = torch.zeros_like(val) if carry is None else carry
            return carry + val + sum(jnt_types)

        b_in = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]])

        out_np = scan.body_tree(
            m_np,
            s_fn,
            "jb",
            "b",
            m_np.jnt_type,
            b_in,
        )
        out_p = scan.body_tree(
            m_p,
            s_fn,
            "jb",
            "b",
            m_p.jnt_type,
            b_in,
        )
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))

        out_np = scan.body_tree(
            m_np,
            s_fn,
            "jb",
            "b",
            m_np.jnt_type,
            b_in,
            reverse=True,
        )
        out_p = scan.body_tree(
            m_p,
            s_fn,
            "jb",
            "b",
            m_p.jnt_type,
            b_in,
            reverse=True,
        )
        np.testing.assert_array_equal(np.array(out_np), np.array(out_p))


class ScanPaddingPhysicsTest(absltest.TestCase):
    """Test padded scans through full physics pipeline."""

    def test_kinematics_equivalence(self):
        from mujoco_torch._src import test_util

        m_mj = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m_mj)

        mx_np = mujoco_torch.device_put(m_mj)
        mx_p = mujoco_torch.device_put(m_mj, scan_padding=True)

        np.random.seed(42)
        d.qvel = np.random.random(m_mj.nv) * 0.05
        mujoco.mj_step(m_mj, d)

        qpos = torch.tensor(d.qpos.copy())
        qvel = torch.tensor(d.qvel.copy())

        dx_np = mujoco_torch.make_data(mx_np)
        dx_p = mujoco_torch.make_data(mx_p)

        dx_np = mujoco_torch.kinematics(
            mx_np,
            dx_np.replace(qpos=qpos, qvel=qvel),
        )
        dx_p = mujoco_torch.kinematics(
            mx_p,
            dx_p.replace(qpos=qpos, qvel=qvel),
        )

        for attr in ("xpos", "xquat", "xmat", "xanchor", "xaxis"):
            np.testing.assert_allclose(
                getattr(dx_np, attr),
                getattr(dx_p, attr),
                atol=1e-12,
                err_msg=f"mismatch in {attr}",
            )

    def test_step_equivalence(self):
        from mujoco_torch._src import forward, test_util

        m_mj = test_util.load_test_file("ant.xml")

        mx_np = mujoco_torch.device_put(m_mj)
        mx_p = mujoco_torch.device_put(m_mj, scan_padding=True)

        d = mujoco.MjData(m_mj)
        np.random.seed(42)
        d.qvel = np.random.random(m_mj.nv) * 0.05
        mujoco.mj_step(m_mj, d)

        qpos = torch.tensor(d.qpos.copy())
        qvel = torch.tensor(d.qvel.copy())

        dx_np = mujoco_torch.make_data(mx_np).replace(
            qpos=qpos,
            qvel=qvel,
        )
        dx_p = mujoco_torch.make_data(mx_p).replace(
            qpos=qpos,
            qvel=qvel,
        )

        for _ in range(3):
            dx_np = forward.step(mx_np, dx_np)
            dx_p = forward.step(mx_p, dx_p)

        np.testing.assert_allclose(
            dx_np.qpos,
            dx_p.qpos,
            atol=1e-10,
            err_msg="qpos diverged after 3 steps",
        )
        np.testing.assert_allclose(
            dx_np.qvel,
            dx_p.qvel,
            atol=1e-10,
            err_msg="qvel diverged after 3 steps",
        )


if __name__ == "__main__":
    absltest.main()
