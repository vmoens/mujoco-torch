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
"""Tests for support."""

import mujoco
import numpy as np

# from torch import numpy as torch
import torch
from absl.testing import absltest, parameterized

import mujoco_torch
from mujoco_torch._src import smooth, support, test_util


class SupportTest(parameterized.TestCase):
    @parameterized.parameters(set(test_util.TEST_FILES) - {"convex.xml"})
    def test_jac(self, fname):
        np.random.seed(0)

        m = test_util.load_test_file(fname)
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        point = torch.tensor(np.random.randn(3))
        body = int(np.random.choice(m.nbody))
        jacp, jacr = support.jac(mx, dx, point, body)

        jacp_expected, jacr_expected = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp_expected, jacr_expected, point, body)
        np.testing.assert_almost_equal(jacp, jacp_expected.T, 6)
        np.testing.assert_almost_equal(jacr, jacr_expected.T, 6)

    def test_xfrc_accumulate(self):
        """Tests that xfrc_accumulate ouput matches mj_xfrcAccumulate."""
        np.random.seed(0)

        m = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m)
        mujoco.mj_step(m, d)
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        self.assertFalse((dx.xipos == 0.0).all())

        xfrc = np.random.rand(*dx.xfrc_applied.shape)

        d.xfrc_applied[:] = xfrc
        dx = dx.replace(xfrc_applied=torch.tensor(xfrc))

        qfrc = support.xfrc_accumulate(mx, dx)
        qfrc_expected = np.zeros(m.nv)
        for i in range(1, m.nbody):
            mujoco.mj_applyFT(
                m,
                d,
                d.xfrc_applied[i, :3],
                d.xfrc_applied[i, 3:],
                d.xipos[i],
                i,
                qfrc_expected,
            )

        np.testing.assert_almost_equal(qfrc, qfrc_expected, 6)

    def test_full_m_sparse_matches_mujoco_under_vmap(self):
        """full_m rebuilds the dense mass matrix of a sparse model, also under vmap.

        Twelve free bodies give 72 degrees of freedom, above the threshold at
        which ``jacobian="auto"`` stores the mass matrix sparse. The reference
        is MuJoCo C's mass matrix applied to the basis vectors.
        """
        m = mujoco.MjModel.from_xml_string(test_util.free_spheres_xml(12))
        self.assertTrue(support.is_sparse(m))
        mx = mujoco_torch.device_put(m)

        rng = np.random.RandomState(0)
        datas, expected = [], []
        for _ in range(3):
            d = mujoco.MjData(m)
            d.qpos[:] = m.qpos0 + 0.1 * rng.randn(m.nq)
            mujoco.mj_forward(m, d)
            full = np.zeros((m.nv, m.nv))
            column = np.zeros(m.nv)
            for k in range(m.nv):
                mujoco.mj_mulM(m, d, column, np.eye(m.nv)[k])
                full[:, k] = column
            expected.append(full)
            # The torch side gets the pose only: crb fills qM in the sparse
            # layout full_m reads, and without the C constraint arrays (whose
            # size differs with the contacts of each pose) the envs stack.
            pose = mujoco.MjData(m)
            pose.qpos[:] = d.qpos
            dx = mujoco_torch.device_put(pose)
            dx = smooth.crb(mx, smooth.com_pos(mx, smooth.kinematics(mx, dx)))
            datas.append(dx)

        np.testing.assert_allclose(support.full_m(mx, datas[0]), expected[0], atol=1e-6)

        batched = torch.vmap(lambda d: support.full_m(mx, d))(torch.stack(datas, dim=0))
        np.testing.assert_allclose(batched, np.stack(expected), atol=1e-6)


if __name__ == "__main__":
    absltest.main()
