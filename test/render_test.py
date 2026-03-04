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
"""Tests for the pure-PyTorch ray-cast renderer."""

import mujoco
import numpy as np
import torch
from absl.testing import absltest

import mujoco_torch

_RENDER_XML = """
<mujoco>
  <visual>
    <global fovy="45"/>
  </visual>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.01" rgba="0.5 0.5 0.5 1"/>
    <body name="ball" pos="0 0 1">
      <joint name="slide" type="slide" axis="0 0 1"/>
      <geom name="sphere" type="sphere" size="0.3" rgba="1 0 0 1"/>
    </body>
    <camera name="cam0" pos="2 0 1" xyaxes="0 1 0 0 0 1"/>
  </worldbody>
</mujoco>
"""

_MULTI_GEOM_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="5 5 0.01" rgba="0.3 0.3 0.3 1"/>
    <body name="box_body" pos="0 0 0.5">
      <joint name="j1" type="slide" axis="0 0 1"/>
      <geom name="box" type="box" size="0.2 0.2 0.2" rgba="0 1 0 1"/>
    </body>
    <body name="capsule_body" pos="1 0 0.5">
      <joint name="j2" type="slide" axis="0 0 1"/>
      <geom name="capsule" type="capsule" size="0.1 0.3" rgba="0 0 1 1"/>
    </body>
    <camera name="cam0" pos="3 0 1" xyaxes="0 1 0 0 0 1"/>
  </worldbody>
</mujoco>
"""


class RenderTest(absltest.TestCase):
    def test_render_returns_correct_shapes(self):
        """render() returns (H, W, 3), (H, W), (H, W) tensors."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        W, H = 16, 12
        rgb, depth, seg = mujoco_torch.render(mx, dx, camera_id=0, width=W, height=H)

        self.assertEqual(rgb.shape, (H, W, 3))
        self.assertEqual(depth.shape, (H, W))
        self.assertEqual(seg.shape, (H, W))
        self.assertTrue(rgb.is_floating_point())
        self.assertTrue(seg.dtype in (torch.int32, torch.int64, torch.long))

    def test_depth_is_positive_on_hits(self):
        """Rays that hit geometry report positive depth."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        _, depth, seg = mujoco_torch.render(mx, dx, camera_id=0, width=32, height=32)

        hit_mask = seg >= 0
        self.assertTrue(hit_mask.any(), "Expected at least some ray hits")
        self.assertTrue((depth[hit_mask] > 0).all())

    def test_miss_pixels_have_neg1(self):
        """Pixels that miss all geometry get depth=-1, seg=-1, rgb=0."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        rgb, depth, seg = mujoco_torch.render(mx, dx, camera_id=0, width=32, height=32)

        miss_mask = seg < 0
        if miss_mask.any():
            np.testing.assert_array_equal(depth[miss_mask].numpy(), -1.0)
            np.testing.assert_array_equal(rgb[miss_mask].numpy(), np.zeros((miss_mask.sum(), 3)))

    def test_sphere_color_lookup_flat(self):
        """Flat-shaded: pixels hitting the red sphere are exactly (1, 0, 0)."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        sphere_id = mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_GEOM, "sphere")

        rgb, _, seg = mujoco_torch.render(mx, dx, camera_id=0, width=32, height=32, shading=False)

        sphere_mask = seg == sphere_id
        self.assertTrue(sphere_mask.any(), "Camera should see the sphere")
        sphere_rgb = rgb[sphere_mask]
        np.testing.assert_allclose(sphere_rgb[:, 0].numpy(), 1.0, atol=1e-5)
        np.testing.assert_allclose(sphere_rgb[:, 1].numpy(), 0.0, atol=1e-5)
        np.testing.assert_allclose(sphere_rgb[:, 2].numpy(), 0.0, atol=1e-5)

    def test_sphere_shading(self):
        """Shaded sphere has dominant red channel and varies across pixels."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        sphere_id = mujoco.mj_name2id(m_mj, mujoco.mjtObj.mjOBJ_GEOM, "sphere")

        rgb, _, seg = mujoco_torch.render(mx, dx, camera_id=0, width=32, height=32, shading=True)

        sphere_mask = seg == sphere_id
        self.assertTrue(sphere_mask.any())
        sphere_rgb = rgb[sphere_mask]
        self.assertGreater(
            sphere_rgb[:, 0].mean().item(),
            sphere_rgb[:, 1].mean().item(),
            "Mean red channel should dominate mean green for a red sphere",
        )
        self.assertGreater(
            sphere_rgb[:, 0].mean().item(),
            sphere_rgb[:, 2].mean().item(),
            "Mean red channel should dominate mean blue for a red sphere",
        )

    def test_segmentation_ids(self):
        """Segmentation buffer contains valid geom ids or -1."""
        m_mj = mujoco.MjModel.from_xml_string(_MULTI_GEOM_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        _, _, seg = mujoco_torch.render(mx, dx, camera_id=0, width=32, height=32)

        seg_np = seg.numpy()
        valid = (seg_np >= -1) & (seg_np < m_mj.ngeom)
        self.assertTrue(valid.all(), f"Invalid geom ids in seg buffer: {np.unique(seg_np)}")

    def test_precompute_render_data_reuse(self):
        """Passing pre-computed render data produces identical output."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)

        precomp = mujoco_torch.precompute_render_data(mx)
        rgb1, depth1, seg1 = mujoco_torch.render(mx, dx, camera_id=0, width=16, height=16)
        rgb2, depth2, seg2 = mujoco_torch.render(mx, dx, camera_id=0, width=16, height=16, precomp=precomp)

        np.testing.assert_array_equal(rgb1.numpy(), rgb2.numpy())
        np.testing.assert_array_equal(depth1.numpy(), depth2.numpy())
        np.testing.assert_array_equal(seg1.numpy(), seg2.numpy())

    def test_cam_xpos_updated_by_forward(self):
        """forward() updates cam_xpos / cam_xmat via cam_bodyid_t."""
        m_mj = mujoco.MjModel.from_xml_string(_RENDER_XML)
        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)

        mx = mujoco_torch.device_put(m_mj)
        dx = mujoco_torch.device_put(d_mj)
        dx = mujoco_torch.forward(mx, dx)

        cam_xpos_np = np.array(d_mj.cam_xpos)
        cam_xpos_torch = dx.cam_xpos.detach().cpu().numpy()
        np.testing.assert_allclose(cam_xpos_torch, cam_xpos_np, atol=1e-6)


if __name__ == "__main__":
    absltest.main()
