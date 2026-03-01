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
"""Tests for constraint functions."""

import mujoco

# pylint: enable=g-importing-member
import numpy as np
import torch
from absl.testing import absltest, parameterized

import mujoco_torch
from mujoco_torch._src import collision_driver, constraint, forward, test_util

# pylint: disable=g-importing-member
from mujoco_torch._src.types import DisableBit, SolverType


def _assert_eq(a, b, name, step, fname, atol=5e-3, rtol=5e-3):
    err_msg = f"mismatch: {name} at step {step} in {fname}"
    np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class ConstraintTest(parameterized.TestCase):
    @parameterized.parameters(enumerate(test_util.TEST_FILES))
    def test_constraints(self, seed, fname):
        """Test constraints."""
        np.random.seed(seed)

        # exclude convex.xml since convex contacts are not exactly equivalent
        if fname == "convex.xml":
            return

        m = test_util.load_test_file(fname)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.make_data(mx)

        forward_jit_fn = mujoco_torch.forward

        # give the system a little kick to ensure we have non-identity rotations
        d.qvel = np.random.random(m.nv)
        for i in range(100):
            dx = dx.replace(qpos=torch.tensor(d.qpos.copy()), qvel=torch.tensor(d.qvel.copy()))
            mujoco.mj_step(m, d)
            dx = forward_jit_fn(mx, dx)

            # Use tolerance-based filter: rows with max abs value > eps
            eps = 1e-10
            nnz_filter = dx.efc_J.abs().max(dim=1).values > eps

            mj_efc_j = d.efc_J.reshape((-1, m.nv))
            mj_nnz = np.abs(mj_efc_j).max(axis=1) > eps
            mj_efc_j_filtered = mj_efc_j[mj_nnz]
            mjx_efc_j = dx.efc_J[nnz_filter]

            # Only compare when either side has significant constraint data
            if mj_efc_j_filtered.shape[0] > 0 or mjx_efc_j.shape[0] > 0:
                _assert_eq(mj_efc_j_filtered, mjx_efc_j, "efc_J", i, fname)

                mjx_efc_d = dx.efc_D[nnz_filter]
                mj_efc_d = d.efc_D[mj_nnz] if mj_nnz.any() else d.efc_D[:0]
                _assert_eq(mj_efc_d, mjx_efc_d, "efc_D", i, fname)

                mjx_efc_aref = dx.efc_aref[nnz_filter]
                mj_efc_aref = d.efc_aref[mj_nnz] if mj_nnz.any() else d.efc_aref[:0]
                _assert_eq(mj_efc_aref, mjx_efc_aref, "efc_aref", i, fname)

                mjx_efc_frictionloss = dx.efc_frictionloss[nnz_filter]
                mj_efc_fl = d.efc_frictionloss[mj_nnz] if mj_nnz.any() else d.efc_frictionloss[:0]
                _assert_eq(
                    mj_efc_fl,
                    mjx_efc_frictionloss,
                    "efc_frictionloss",
                    i,
                    fname,
                )

    _JNT_RANGE = """
    <mujoco>
      <worldbody>
        <body pos="0 0 1">
          <joint type="slide" axis="1 0 0" range="-1.8 1.8" solreflimit=".08 1"
           damping="5e-4"/>
          <geom type="box" size="0.2 0.15 0.1" mass="1"/>
          <body>
            <joint axis="0 1 0" damping="2e-6"/>
            <geom type="capsule" fromto="0 0 0 0 0 1" size="0.045" mass=".1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  """

    def test_jnt_range(self):
        """Tests that mixed joint ranges are respected."""
        # TODO(robotics-simulation): also test ball
        m = mujoco.MjModel.from_xml_string(self._JNT_RANGE)
        m.opt.solver = SolverType.CG.value
        d = mujoco.MjData(m)
        d.qpos = np.array([2.0, 15.0])

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        precomp = mx._device_precomp["constraint_data_py"]["limit_slide_hinge"]
        self.assertIsNotNone(precomp)
        efc = constraint._instantiate_limit_slide_hinge(mx, dx, precomp)

        # first joint is outside the joint range
        np.testing.assert_array_almost_equal(efc.J[0, 0], -1.0)

        # second joint has no range, so only one efc row
        self.assertEqual(efc.J.shape[0], 1)

    def test_disable_refsafe(self):
        m = test_util.load_test_file("ant.xml")

        timeconst = m.opt.timestep / 4.0  # timeconst < 2 * timestep
        solimp = torch.tensor([timeconst, 1.0], dtype=torch.float64)
        solref = torch.tensor([0.8, 0.99, 0.001, 0.2, 2], dtype=torch.float64)
        pos = torch.ones(3, dtype=torch.float64)

        m.opt.disableflags = m.opt.disableflags | DisableBit.REFSAFE
        mx = mujoco_torch.device_put(m)
        refsafe = mx.constraint_data_py["refsafe"]
        k, *_ = constraint._kbi(mx, solimp, solref, pos, refsafe=refsafe)
        self.assertEqual(k, 1 / (0.99**2 * timeconst**2))

        m.opt.disableflags = m.opt.disableflags & ~DisableBit.REFSAFE
        mx = mujoco_torch.device_put(m)
        refsafe = mx.constraint_data_py["refsafe"]
        k, *_ = constraint._kbi(mx, solimp, solref, pos, refsafe=refsafe)
        self.assertEqual(k, 1 / (0.99**2 * (2 * m.opt.timestep) ** 2))

    def test_disableconstraint(self):
        m = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m)

        m.opt.disableflags = m.opt.disableflags & ~DisableBit.CONSTRAINT
        mx, dx = mujoco_torch.device_put(m), mujoco_torch.device_put(d)
        dx = constraint.make_constraint(mx, dx)
        self.assertGreater(dx.efc_J.shape[0], 1)

        m.opt.disableflags = m.opt.disableflags | DisableBit.CONSTRAINT
        mx = mujoco_torch.device_put(m)
        dx = constraint.make_constraint(mx, dx)
        self.assertEqual(dx.efc_J.shape[0], 0)

    def test_disable_equality(self):
        m = test_util.load_test_file("equality.xml")
        d = mujoco.MjData(m)

        m.opt.disableflags = m.opt.disableflags | DisableBit.EQUALITY
        mx, dx = mujoco_torch.device_put(m), mujoco_torch.device_put(d)
        dx = constraint.make_constraint(mx, dx)
        self.assertEqual(dx.efc_J.shape[0], 0)

    def test_disable_contact(self):
        m = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m)
        d.qpos[2] = 0.0
        mujoco.mj_forward(m, d)

        m.opt.disableflags = m.opt.disableflags & ~DisableBit.CONTACT
        mx, dx = mujoco_torch.device_put(m), mujoco_torch.device_put(d)
        dx = dx.tree_replace({"contact.frame": dx.contact.frame.reshape((-1, 3, 3))})
        ncon_fl, ncon_fr = mx.condim_counts_py
        self.assertGreater(ncon_fr, 0)
        efc = constraint._instantiate_contact(mx, dx)
        self.assertIsNotNone(efc)

        m.opt.disableflags = m.opt.disableflags | DisableBit.CONTACT
        mx = mujoco_torch.device_put(m)
        ncon_fl, ncon_fr = mx.condim_counts_py
        self.assertEqual(ncon_fr, 0)

    _FRICTIONLOSS_XML = """
    <mujoco>
      <option solver="CG" timestep="0.005"/>
      <worldbody>
        <body pos="0 0 1">
          <joint type="hinge" axis="0 0 1" frictionloss="1.0" damping="0.1"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body pos="0.5 0 0">
            <joint type="slide" axis="1 0 0" frictionloss="0.5" damping="0.1"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body>
        </body>
      </worldbody>
      <tendon>
        <fixed name="ten0">
          <joint joint="joint0" coef="1"/>
        </fixed>
      </tendon>
    </mujoco>
    """

    _FRICTIONLOSS_DOF_ONLY_XML = """
    <mujoco>
      <option solver="CG" timestep="0.005"/>
      <worldbody>
        <body pos="0 0 1">
          <joint type="hinge" axis="0 0 1" frictionloss="2.0" damping="0.1"/>
          <geom type="sphere" size="0.1" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def test_frictionloss_dof(self):
        """Test DOF frictionloss constraints match MuJoCo C."""
        m = mujoco.MjModel.from_xml_string(self._FRICTIONLOSS_DOF_ONLY_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[:] = np.array([1.0])
        for i in range(100):
            dx = mujoco_torch.device_put(d)
            mujoco.mj_step(m, d)
            dx = mujoco_torch.forward(mx, dx)

            eps = 1e-10
            nnz_filter = dx.efc_J.abs().max(dim=1).values > eps
            mj_efc_j = d.efc_J.reshape((-1, m.nv))
            mj_nnz = np.abs(mj_efc_j).max(axis=1) > eps

            if mj_nnz.any() or nnz_filter.any():
                _assert_eq(mj_efc_j[mj_nnz], dx.efc_J[nnz_filter], "efc_J", i, "frictionloss_dof")
                _assert_eq(d.efc_D[mj_nnz], dx.efc_D[nnz_filter], "efc_D", i, "frictionloss_dof")
                _assert_eq(d.efc_aref[mj_nnz], dx.efc_aref[nnz_filter], "efc_aref", i, "frictionloss_dof")
                _assert_eq(
                    d.efc_frictionloss[mj_nnz],
                    dx.efc_frictionloss[nnz_filter],
                    "efc_frictionloss",
                    i,
                    "frictionloss_dof",
                )

    def test_frictionloss_instantiate(self):
        """Test that _instantiate_friction creates the right rows."""
        m = mujoco.MjModel.from_xml_string(self._FRICTIONLOSS_DOF_ONLY_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        precomp = mx._device_precomp["constraint_data_py"]["friction"]
        self.assertIsNotNone(precomp)
        efc = constraint._instantiate_friction(mx, dx, precomp)

        self.assertEqual(efc.J.shape[0], 1)
        np.testing.assert_array_almost_equal(efc.frictionloss.numpy(), [2.0])
        np.testing.assert_array_almost_equal(efc.pos.numpy(), [0.0])

    def test_disable_frictionloss(self):
        """Test that FRICTIONLOSS disable flag suppresses friction rows."""
        m = mujoco.MjModel.from_xml_string(self._FRICTIONLOSS_DOF_ONLY_XML)
        m.opt.disableflags = m.opt.disableflags | DisableBit.FRICTIONLOSS
        mx = mujoco_torch.device_put(m)
        self.assertIsNone(mx.constraint_data_py["friction"])

    _CONDIM1_XML = """
    <mujoco>
      <option solver="CG" timestep="0.005"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1" condim="1"/>
        <body pos="0 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1" condim="1"/>
        </body>
      </worldbody>
    </mujoco>
    """

    _MIXED_CONDIM_XML = """
    <mujoco>
      <option solver="CG" timestep="0.005"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1" condim="1"/>
        <body name="ball_fl" pos="-0.5 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1" condim="1"/>
        </body>
        <body name="ball_fr" pos="0.5 0 1">
          <freejoint/>
          <geom type="sphere" size="0.1" mass="1" condim="3"/>
        </body>
      </worldbody>
    </mujoco>
    """

    def test_condim1_device_put(self):
        """Verify that device_put accepts a model with condim=1."""
        m = mujoco.MjModel.from_xml_string(self._CONDIM1_XML)
        mx = mujoco_torch.device_put(m)
        self.assertIsNotNone(mx)

    def test_condim1_frictionless_step(self):
        """Step a condim=1 model and compare qpos/qvel against MuJoCo C."""
        m = mujoco.MjModel.from_xml_string(self._CONDIM1_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[:3] = [0.1, 0.0, -0.5]
        for i in range(200):
            dx = mujoco_torch.device_put(d)
            mujoco.mj_step(m, d)
            dx = mujoco_torch.step(mx, dx)

            _assert_eq(d.qpos, dx.qpos, "qpos", i, "condim1")
            _assert_eq(d.qvel, dx.qvel, "qvel", i, "condim1")

    def test_condim1_normal_only(self):
        """Verify condim=1 constraints produce only normal-force rows."""
        m = mujoco.MjModel.from_xml_string(self._CONDIM1_XML)
        d = mujoco.MjData(m)
        d.qpos[2] = 0.05  # push sphere close to plane
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        ncon_fl, _ = mx.condim_counts_py
        if ncon_fl > 0:
            efc = constraint._instantiate_contact_frictionless(mx, dx)
            ncon = dx.contact.dist.shape[0]
            self.assertEqual(efc.J.shape[0], ncon)

    def test_mixed_condim_step(self):
        """Step a model with both condim=1 and condim=3, compare against C."""
        m = mujoco.MjModel.from_xml_string(self._MIXED_CONDIM_XML)
        d = mujoco.MjData(m)
        mx = mujoco_torch.device_put(m)

        d.qvel[2] = -0.5
        d.qvel[8] = -0.5
        for i in range(200):
            dx = mujoco_torch.device_put(d)
            mujoco.mj_step(m, d)
            dx = mujoco_torch.step(mx, dx)

            _assert_eq(d.qpos, dx.qpos, "qpos", i, "mixed_condim")
            _assert_eq(d.qvel, dx.qvel, "qvel", i, "mixed_condim")

    def test_mixed_condim_constraint_sizes(self):
        """Verify constraint sizes account for variable rows per condim."""
        m = mujoco.MjModel.from_xml_string(self._MIXED_CONDIM_XML)
        mx = mujoco_torch.device_put(m)
        ne, nf, nl, ncon_, nefc = constraint.constraint_sizes(mx)

        dims = collision_driver.make_condim(mx)
        ncon_fl = int((dims == 1).sum())
        ncon_fr = int((dims == 3).sum())

        self.assertEqual(ncon_, ncon_fl + ncon_fr)
        expected_nc = ncon_fl * 1 + ncon_fr * 4
        self.assertEqual(nefc, ne + nf + nl + expected_nc)

    def test_make_constraint_no_graph_breaks_contacts(self):
        """make_constraint compiles with fullgraph=True for contact models."""
        m = test_util.load_test_file("ant.xml")
        d = mujoco.MjData(m)
        d.qpos[2] = 0.0
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = forward._position(mx, dx)

        compiled_fn = torch.compile(constraint.make_constraint, fullgraph=True, backend="aot_eager")
        dx_compiled = compiled_fn(mx, dx)
        dx_eager = constraint.make_constraint(mx, dx)

        np.testing.assert_allclose(
            dx_compiled.efc_J.detach().numpy(),
            dx_eager.efc_J.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_J mismatch (contacts)",
        )
        np.testing.assert_allclose(
            dx_compiled.efc_D.detach().numpy(),
            dx_eager.efc_D.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_D mismatch (contacts)",
        )
        np.testing.assert_allclose(
            dx_compiled.efc_aref.detach().numpy(),
            dx_eager.efc_aref.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_aref mismatch (contacts)",
        )

    def test_make_constraint_no_graph_breaks_equality(self):
        """make_constraint compiles with fullgraph=True for equality models."""
        m = test_util.load_test_file("equality.xml")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = forward._position(mx, dx)

        compiled_fn = torch.compile(constraint.make_constraint, fullgraph=True, backend="aot_eager")
        dx_compiled = compiled_fn(mx, dx)
        dx_eager = constraint.make_constraint(mx, dx)

        np.testing.assert_allclose(
            dx_compiled.efc_J.detach().numpy(),
            dx_eager.efc_J.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_J mismatch (equality)",
        )

    def test_make_constraint_no_graph_breaks_frictionloss(self):
        """make_constraint compiles with fullgraph=True for frictionloss models."""
        m = mujoco.MjModel.from_xml_string(self._FRICTIONLOSS_DOF_ONLY_XML)
        d = mujoco.MjData(m)
        d.qvel[:] = 1.0
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = forward._position(mx, dx)

        compiled_fn = torch.compile(constraint.make_constraint, fullgraph=True, backend="aot_eager")
        dx_compiled = compiled_fn(mx, dx)
        dx_eager = constraint.make_constraint(mx, dx)

        np.testing.assert_allclose(
            dx_compiled.efc_J.detach().numpy(),
            dx_eager.efc_J.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_J mismatch (frictionloss)",
        )
        np.testing.assert_allclose(
            dx_compiled.efc_frictionloss.detach().numpy(),
            dx_eager.efc_frictionloss.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_frictionloss mismatch",
        )

    def test_make_constraint_no_graph_breaks_mixed_condim(self):
        """make_constraint compiles with fullgraph=True for mixed condim models."""
        m = mujoco.MjModel.from_xml_string(self._MIXED_CONDIM_XML)
        d = mujoco.MjData(m)
        d.qpos[2] = 0.05
        d.qpos[9] = 0.05
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = forward._position(mx, dx)

        compiled_fn = torch.compile(constraint.make_constraint, fullgraph=True, backend="aot_eager")
        dx_compiled = compiled_fn(mx, dx)
        dx_eager = constraint.make_constraint(mx, dx)

        np.testing.assert_allclose(
            dx_compiled.efc_J.detach().numpy(),
            dx_eager.efc_J.detach().numpy(),
            atol=1e-6,
            err_msg="compiled make_constraint efc_J mismatch (mixed condim)",
        )

    def test_make_constraint_no_graph_breaks_disabled(self):
        """make_constraint compiles with fullgraph=True when constraints disabled."""
        m = test_util.load_test_file("ant.xml")
        m.opt.disableflags = m.opt.disableflags | DisableBit.CONSTRAINT
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        mx = mujoco_torch.device_put(m)
        dx = mujoco_torch.device_put(d)
        dx = forward._position(mx, dx)

        compiled_fn = torch.compile(constraint.make_constraint, fullgraph=True, backend="aot_eager")
        dx_compiled = compiled_fn(mx, dx)

        self.assertEqual(dx_compiled.efc_J.shape[0], 0)


if __name__ == "__main__":
    absltest.main()
