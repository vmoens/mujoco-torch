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
"""Tests for sensor functions."""

import mujoco
import numpy as np
import torch
from absl.testing import absltest, parameterized

import mujoco_torch
from mujoco_torch._src import forward, sensor, test_util
from mujoco_torch._src.types import SensorType as ST


_SUPPORTED_SENSOR_TYPES = {
    int(ST.MAGNETOMETER), int(ST.RANGEFINDER), int(ST.JOINTPOS),
    int(ST.TENDONPOS), int(ST.ACTUATORPOS), int(ST.BALLQUAT),
    int(ST.FRAMEPOS), int(ST.FRAMEXAXIS), int(ST.FRAMEYAXIS),
    int(ST.FRAMEZAXIS), int(ST.FRAMEQUAT), int(ST.SUBTREECOM),
    int(ST.CLOCK), int(ST.VELOCIMETER), int(ST.GYRO),
    int(ST.JOINTVEL), int(ST.TENDONVEL), int(ST.ACTUATORVEL),
    int(ST.BALLANGVEL), int(ST.SUBTREELINVEL), int(ST.SUBTREEANGMOM),
    int(ST.ACCELEROMETER), int(ST.FORCE), int(ST.TORQUE),
    int(ST.ACTUATORFRC), int(ST.JOINTACTFRC), int(ST.TENDONACTFRC),
}


def _supported_mask(m):
    """Boolean mask over sensordata for supported sensor types."""
    mask = np.zeros(m.nsensordata, dtype=bool)
    for i in range(m.nsensor):
        if int(m.sensor_type[i]) in _SUPPORTED_SENSOR_TYPES:
            adr = int(m.sensor_adr[i])
            dim = int(m.sensor_dim[i])
            mask[adr:adr + dim] = True
    return mask


_SENSOR_XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="5 5 0.01"/>
    <body name="body1" pos="0 0 1">
      <joint name="hinge1" type="hinge" axis="0 1 0"/>
      <geom name="geom1" type="sphere" size="0.1" mass="1"/>
      <site name="site1" pos="0.1 0 0"/>
      <body name="body2" pos="0.5 0 0">
        <joint name="slide1" type="slide" axis="1 0 0"/>
        <geom name="geom2" type="sphere" size="0.1" mass="1"/>
        <site name="site2" pos="0.1 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="hinge1" gear="10"/>
    <motor name="motor2" joint="slide1" gear="10"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge1"/>
    <jointpos joint="slide1"/>
    <jointvel joint="hinge1"/>
    <jointvel joint="slide1"/>
    <framepos objtype="site" objname="site1"/>
    <subtreecom body="body1"/>
    <velocimeter site="site1"/>
    <gyro site="site1"/>
    <magnetometer site="site1"/>
    <clock/>
  </sensor>
</mujoco>
"""

_FRAMEPOS_XML = """
<mujoco>
  <option timestep="0.005">
    <flag constraint="disable"/>
  </option>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size="0.1" mass="1"/>
      <site name="s1" pos="0.1 0 0"/>
      <site name="s_ref" pos="0 0.1 0"/>
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="site" objname="s1"/>
    <framepos objtype="site" objname="s1"
              reftype="site" refname="s_ref"/>
    <framexaxis objtype="site" objname="s1"/>
    <frameyaxis objtype="site" objname="s1"/>
    <framezaxis objtype="site" objname="s1"/>
  </sensor>
</mujoco>
"""


def _run_forward(m, d, mx, dx):
    """Run MuJoCo and MJX forward, return (mj_data, mjx_data)."""
    mujoco.mj_forward(m, d)
    dx = forward.forward(mx, dx)
    return d, dx


class SensorTest(parameterized.TestCase):

    def test_sensor_pos_vel_acc(self):
        """Position/velocity/acceleration sensors match MuJoCo."""
        m = mujoco.MjModel.from_xml_string(_SENSOR_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[:] = np.array([0.5, -0.3])
        d.ctrl[:] = np.array([1.0, -0.5])
        qpos = torch.tensor(d.qpos.copy())
        qvel = torch.tensor(d.qvel.copy())
        dx = mujoco_torch.device_put(d)
        dx = dx.replace(qpos=qpos, qvel=qvel)

        d, dx = _run_forward(m, d, mx, dx)

        mask = _supported_mask(m)
        np.testing.assert_allclose(
            dx.sensordata.detach().numpy()[mask],
            d.sensordata[:mx.nsensordata][mask],
            atol=1e-3,
            rtol=1e-3,
            err_msg="sensordata mismatch",
        )

    def test_sensor_framepos(self):
        """FRAMEPOS / FRAMEAXIS sensors match MuJoCo."""
        m = mujoco.MjModel.from_xml_string(_FRAMEPOS_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[:] = 0.3
        qpos = torch.tensor(d.qpos.copy())
        qvel = torch.tensor(d.qvel.copy())
        dx = mujoco_torch.device_put(d)
        dx = dx.replace(qpos=qpos, qvel=qvel)

        d, dx = _run_forward(m, d, mx, dx)

        mask = _supported_mask(m)
        np.testing.assert_allclose(
            dx.sensordata.detach().numpy()[mask],
            d.sensordata[:mx.nsensordata][mask],
            atol=1e-3,
            rtol=1e-3,
            err_msg="framepos/axis sensordata mismatch",
        )

    @parameterized.parameters(
        f for f in test_util.TEST_FILES
        if f not in ("equality.xml",)
    )
    def test_sensor_step(self, fname):
        """Sensors are correct after multiple steps."""
        m = test_util.load_test_file(fname)
        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        d.qvel[:] = np.random.RandomState(42).randn(m.nv) * 0.05
        mask = _supported_mask(m)
        if not mask.any():
            return

        for i in range(10):
            qpos, qvel = d.qpos.copy(), d.qvel.copy()
            d2 = mujoco.MjData(m)
            d2.qpos[:] = qpos
            d2.qvel[:] = qvel
            dx = mujoco_torch.device_put(d2)

            mujoco.mj_step(m, d)
            dx = forward.step(mx, dx)

            np.testing.assert_allclose(
                dx.sensordata.detach().numpy()[mask],
                d.sensordata[:mx.nsensordata][mask],
                atol=5e-2,
                rtol=5e-2,
                err_msg=(
                    f"sensordata mismatch at step {i} "
                    f"in {fname}"
                ),
            )

    def test_sensor_no_graph_breaks(self):
        """sensor_pos/vel/acc compile with fullgraph=True."""
        m = mujoco.MjModel.from_xml_string(_SENSOR_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        d.qvel[:] = np.array([0.5, -0.3])
        dx = mujoco_torch.device_put(d)
        dx = forward.forward(mx, dx)

        pos_fn = torch.compile(
            sensor.sensor_pos, fullgraph=True,
        )
        vel_fn = torch.compile(
            sensor.sensor_vel, fullgraph=True,
        )
        acc_fn = torch.compile(
            sensor.sensor_acc, fullgraph=True,
        )

        dx_pos = pos_fn(mx, dx)
        dx_vel = vel_fn(mx, dx)
        dx_acc = acc_fn(mx, dx)

        ref_pos = sensor.sensor_pos(mx, dx)
        ref_vel = sensor.sensor_vel(mx, dx)
        ref_acc = sensor.sensor_acc(mx, dx)

        np.testing.assert_allclose(
            dx_pos.sensordata.detach().numpy(),
            ref_pos.sensordata.detach().numpy(),
            atol=1e-6,
            err_msg="compiled sensor_pos mismatch",
        )
        np.testing.assert_allclose(
            dx_vel.sensordata.detach().numpy(),
            ref_vel.sensordata.detach().numpy(),
            atol=1e-6,
            err_msg="compiled sensor_vel mismatch",
        )
        np.testing.assert_allclose(
            dx_acc.sensordata.detach().numpy(),
            ref_acc.sensordata.detach().numpy(),
            atol=1e-6,
            err_msg="compiled sensor_acc mismatch",
        )

    def test_sensor_compile_framepos(self):
        """FRAMEPOS/FRAMEAXIS compile without graph breaks."""
        m = mujoco.MjModel.from_xml_string(_FRAMEPOS_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        d.qvel[:] = 0.3
        dx = mujoco_torch.device_put(d)
        dx = forward.forward(mx, dx)

        pos_fn = torch.compile(
            sensor.sensor_pos, fullgraph=True,
        )
        dx_pos = pos_fn(mx, dx)
        ref_pos = sensor.sensor_pos(mx, dx)

        np.testing.assert_allclose(
            dx_pos.sensordata.detach().numpy(),
            ref_pos.sensordata.detach().numpy(),
            atol=1e-6,
            err_msg="compiled framepos sensor_pos mismatch",
        )

    def test_sensor_ant(self):
        """Ant sensors (rangefinder, velocimeter, gyro, accel)."""
        m = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[:] = (
            np.random.RandomState(7).randn(m.nv) * 0.05
        )
        d.ctrl[:] = (
            np.random.RandomState(8).randn(m.nu) * 0.1
        )
        qpos = torch.tensor(d.qpos.copy())
        qvel = torch.tensor(d.qvel.copy())
        dx = mujoco_torch.device_put(d)
        dx = dx.replace(qpos=qpos, qvel=qvel)

        d, dx = _run_forward(m, d, mx, dx)

        mask = _supported_mask(m)
        np.testing.assert_allclose(
            dx.sensordata.detach().numpy()[mask],
            d.sensordata[:mx.nsensordata][mask],
            atol=1e-2,
            rtol=1e-2,
            err_msg="ant sensordata mismatch",
        )

    def test_sensor_empty_model(self):
        """Models with no sensors produce empty groups."""
        xml = """
        <mujoco>
          <worldbody>
            <body pos="0 0 1">
              <joint type="hinge"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        m = mujoco.MjModel.from_xml_string(xml)
        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        dx = mujoco_torch.device_put(d)
        dx = forward.forward(mx, dx)

        self.assertEqual(len(mx.sensor_groups_pos_py), 0)
        self.assertEqual(len(mx.sensor_groups_vel_py), 0)
        self.assertEqual(len(mx.sensor_groups_acc_py), 0)

    def test_sensor_groups_precomputed(self):
        """Pre-computed groups contain expected sensor types."""
        m = mujoco.MjModel.from_xml_string(_SENSOR_XML)
        mx = mujoco_torch.device_put(m)

        pos_types = {g["type"] for g in mx.sensor_groups_pos_py}
        vel_types = {g["type"] for g in mx.sensor_groups_vel_py}
        acc_types = {g["type"] for g in mx.sensor_groups_acc_py}

        self.assertIn(int(ST.JOINTPOS), pos_types)
        self.assertIn(int(ST.FRAMEPOS), pos_types)
        self.assertIn(int(ST.SUBTREECOM), pos_types)
        self.assertIn(int(ST.MAGNETOMETER), pos_types)
        self.assertIn(int(ST.CLOCK), pos_types)

        self.assertIn(int(ST.JOINTVEL), vel_types)
        self.assertIn(int(ST.VELOCIMETER), vel_types)
        self.assertIn(int(ST.GYRO), vel_types)

        self.assertEqual(len(acc_types), 0)


if __name__ == "__main__":
    absltest.main()
