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
from absl.testing import absltest
from absl.testing import parameterized
import mujoco_torch
from mujoco_torch._src import test_util


def _assert_attr_eq(a, b, attr, fname, atol=1e-3, rtol=1e-3):
  err_msg = f'mismatch: {attr} in {fname}'
  a, b = getattr(a, attr), getattr(b, attr)
  np.testing.assert_allclose(a, b, err_msg=err_msg, atol=atol, rtol=rtol)


class InverseTest(parameterized.TestCase):

  @parameterized.parameters(
      filter(lambda s: s not in ('equality.xml',), test_util.TEST_FILES)
  )
  def test_inverse(self, fname):
    """Test mujoco_torch.inverse matches mujoco.mj_inverse."""
    np.random.seed(test_util.TEST_FILES.index(fname))

    m = test_util.load_test_file(fname)
    d = mujoco.MjData(m)

    # step forward to generate a state with contacts
    d.qvel = np.random.random(m.nv) * 0.05
    for _ in range(10):
      mujoco.mj_step(m, d)

    # run C inverse dynamics
    mujoco.mj_inverse(m, d)
    qfrc_inverse_c = d.qfrc_inverse.copy()

    # run torch inverse dynamics from the same state
    mx = mujoco_torch.device_put(m)
    dx = mujoco_torch.device_put(d)
    dx = mujoco_torch.inverse(mx, dx)

    np.testing.assert_allclose(
        dx.qfrc_inverse, qfrc_inverse_c,
        err_msg=f'qfrc_inverse mismatch in {fname}',
        atol=1e-3, rtol=1e-3,
    )


if __name__ == '__main__':
  absltest.main()
