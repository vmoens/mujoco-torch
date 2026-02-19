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
"""Tests passive forces."""

import itertools

import mujoco
import numpy as np
import torch
from absl.testing import absltest, parameterized
from etils import epath

import mujoco_torch
from mujoco_torch._src.types import DisableBit


def _assert_attr_eq(a, b, attr, step, fname, atol=1e-4, rtol=1e-4):
    err_msg = f"mismatch: {attr} at step {step} in {fname}"
    a, b = getattr(a, attr), getattr(b, attr)
    np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class PassiveTest(parameterized.TestCase):
    @parameterized.parameters(enumerate(("ant.xml", "pendula.xml")))
    def test_stiffness_damping(self, seed, fname):
        """Tests stiffness and damping on Ant."""
        np.random.seed(seed)
        path = epath.resource_path("mujoco_torch") / "test_data"
        path /= fname
        m = mujoco.MjModel.from_xml_string(path.read_text())

        # set stiffness/damping
        m.jnt_stiffness = np.random.uniform(size=m.njnt)
        m.dof_damping = np.random.uniform(size=m.nv)
        d = mujoco.MjData(m)
        d.qvel = np.random.random(m.nv)  # random kick

        mx = mujoco_torch.device_put(m)

        passive_jit_fn = mujoco_torch.passive

        for i in range(100):
            qpos, qvel = torch.tensor(d.qpos.copy()), torch.tensor(d.qvel.copy())
            mujoco.mj_step(m, d)
            dx = mujoco_torch.device_put(d)
            dx = passive_jit_fn(mx, dx.replace(qpos=qpos, qvel=qvel))
            _assert_attr_eq(d, dx, "qfrc_passive", i, fname)

    @parameterized.parameters(itertools.product(range(3), ("pendula.xml",)))
    def test_fluid(self, seed, fname):
        np.random.seed(seed)
        path = epath.resource_path("mujoco_torch") / "test_data"
        path /= fname
        m = mujoco.MjModel.from_xml_string(path.read_text())

        # set density/viscosity/wind
        m.opt.density = np.random.uniform()
        m.opt.viscosity = np.random.uniform()
        m.opt.wind = np.random.uniform()

        passive_jit_fn = mujoco_torch.passive

        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        d.qvel = np.random.random(m.nv)  # random kick

        for i in range(100):
            mujoco.mj_step(m, d)
            dx = mujoco_torch.device_put(d)
            mujoco.mj_passive(m, d)
            dx = passive_jit_fn(mx, dx)
            _assert_attr_eq(d, dx, "qfrc_passive", i, fname)

    def test_disable_passive(self):
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option density="1" viscosity="2" wind="0.1 0.2 0.3"/>
          <worldbody>
            <body>
              <joint damping="1" axis="1 0 0" type="ball"/>
              <geom pos="0 0.5 0" size=".15" mass="1" type="sphere"/>
            </body>
          </worldbody>
        </mujoco>
        """)
        m.opt.disableflags |= DisableBit.SPRING | DisableBit.DAMPER
        mx = mujoco_torch.device_put(m)
        d = mujoco.MjData(m)
        dx = mujoco_torch.device_put(d)
        dx = dx.replace(qvel=torch.ones(mx.nv))

        passive_jit_fn = mujoco_torch.passive
        dx = passive_jit_fn(mx, dx)
        np.testing.assert_equal(dx.qfrc_passive, np.zeros(mx.nv))


if __name__ == "__main__":
    absltest.main()
