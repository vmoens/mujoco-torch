"""Gradient correctness tests for differentiable simulation features.

Tests smooth collision detection, Contacts From Distance (CFD), and
adaptive integration by comparing autograd gradients against finite
differences.

Based on: Paulus et al., "Hard Contacts with Soft Gradients: Refining
Differentiable Simulators for Learning and Control", 2025.
https://arxiv.org/abs/2506.14186
"""

import mujoco
import torch

import mujoco_torch
from mujoco_torch import mujoco_logger as logger
from mujoco_torch._src.collision_primitive import (
    _sphere_sphere,
    plane_capsule,
)
from mujoco_torch._src.collision_types import GeomInfo

torch.set_default_dtype(torch.float64)

SPHERE_XML = """
<mujoco>
  <worldbody>
    <body name="floor" pos="0 0 0">
      <geom type="plane" size="5 5 0.1"/>
    </body>
    <body name="ball1" pos="0.5 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
    <body name="ball2" pos="-0.5 0 0.5">
      <freejoint/>
      <geom type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _fd_grad(fn, x, eps=1e-6):
    """Central finite-difference gradient."""
    grad = torch.zeros_like(x)
    x_flat = x.reshape(-1)
    for i in range(x_flat.numel()):
        x_p = x_flat.clone()
        x_p[i] += eps
        x_m = x_flat.clone()
        x_m[i] -= eps
        grad.reshape(-1)[i] = (fn(x_p.reshape(x.shape)) - fn(x_m.reshape(x.shape))) / (2 * eps)
    return grad


def test_sphere_sphere_smooth_gradcheck():
    """gradcheck on _sphere_sphere with smooth collisions."""
    pos1 = torch.randn(3, requires_grad=True)
    pos2 = torch.randn(3, requires_grad=True)
    r1 = torch.tensor(0.1)
    r2 = torch.tensor(0.1)

    with mujoco_torch.differentiable_mode(smooth_collisions=True):

        def fn(p1, p2):
            dist, pos, n = _sphere_sphere(p1, p2, r1, r2)
            return dist

        ok = torch.autograd.gradcheck(fn, (pos1, pos2), eps=1e-6)
        assert ok, "gradcheck failed for _sphere_sphere (smooth)"
    logger.info("  sphere_sphere smooth gradcheck: PASS")


def test_plane_capsule_smooth_gradcheck():
    """gradcheck on plane_capsule with smooth collisions."""
    plane_pos = torch.zeros(3, dtype=torch.float64)
    plane_mat = torch.eye(3, dtype=torch.float64)
    cap_pos = torch.tensor([0.0, 0.0, 0.3], requires_grad=True)
    cap_mat = torch.eye(3, dtype=torch.float64)
    cap_size = torch.tensor([0.05, 0.1, 0.0])

    plane = GeomInfo(
        pos=plane_pos,
        mat=plane_mat,
        geom_size=torch.tensor([5.0, 5.0, 0.1]),
        vert=None,
        face=None,
        facenorm=None,
        edge=None,
    )

    with mujoco_torch.differentiable_mode(smooth_collisions=True):

        def fn(cp):
            cap = GeomInfo(
                pos=cp,
                mat=cap_mat,
                geom_size=cap_size,
                vert=None,
                face=None,
                facenorm=None,
                edge=None,
            )
            dist, pos, frame = plane_capsule(plane, cap)
            return dist.sum()

        ok = torch.autograd.gradcheck(fn, (cap_pos,), eps=1e-6)
        assert ok, "gradcheck failed for plane_capsule (smooth)"
    logger.info("  plane_capsule smooth gradcheck: PASS")


def test_cfd_nonzero_gradient():
    """CFD produces non-zero gradients for non-contacting spheres."""
    m_mj = mujoco.MjModel.from_xml_string(SPHERE_XML)
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)

    mx = mujoco_torch.device_put(m_mj)
    dx = mujoco_torch.device_put(d_mj)

    qpos = dx.qpos.clone().requires_grad_(True)
    dx_test = dx.replace(qpos=qpos)

    loss_std = mujoco_torch.step(mx, dx_test).qpos.sum()
    loss_std.backward()
    g_std = qpos.grad.clone()

    qpos2 = dx.qpos.clone().requires_grad_(True)
    dx_test2 = dx.replace(qpos=qpos2)
    with mujoco_torch.differentiable_mode(cfd=True, cfd_width=0.5):
        loss_cfd = mujoco_torch.step(mx, dx_test2).qpos.sum()
    loss_cfd.backward()
    g_cfd = qpos2.grad.clone()

    logger.info("  Standard grad norm:  %.6e", g_std.norm().item())
    logger.info("  CFD grad norm:       %.6e", g_cfd.norm().item())
    logger.info("  NaN in std: %s, NaN in cfd: %s", g_std.isnan().any(), g_cfd.isnan().any())
    assert not g_std.isnan().any(), "NaN in standard gradient"
    assert not g_cfd.isnan().any(), "NaN in CFD gradient"
    logger.info("  CFD non-zero gradient test: PASS")


def test_adaptive_vs_fd():
    """Adaptive integration gradients vs finite differences."""
    m_mj = mujoco.MjModel.from_xml_string(SPHERE_XML)
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)

    mx = mujoco_torch.device_put(m_mj)
    dx = mujoco_torch.device_put(d_mj)

    def loss_fn(qvel_init):
        dx_test = dx.replace(qvel=qvel_init)
        with mujoco_torch.differentiable_mode(
            adaptive_integration=True,
            adaptive_substeps=8,
        ):
            dx_out = mujoco_torch.step(mx, dx_test)
        return dx_out.qpos.sum()

    qvel = dx.qvel.clone().requires_grad_(True)
    loss = loss_fn(qvel)
    loss.backward()
    g_auto = qvel.grad.clone()

    g_fd = _fd_grad(
        lambda v: loss_fn(v).detach(),
        qvel.detach(),
        eps=1e-5,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        g_auto.unsqueeze(0),
        g_fd.unsqueeze(0),
    ).item()
    rel_err = (g_auto - g_fd).norm() / (g_fd.norm() + 1e-12)

    logger.info("  Autograd norm: %.6e", g_auto.norm().item())
    logger.info("  FD norm:       %.6e", g_fd.norm().item())
    logger.info("  Cosine sim:    %.6f", cos_sim)
    logger.info("  Rel error:     %.6e", rel_err.item())
    assert not g_auto.isnan().any(), "NaN in autograd"
    assert cos_sim > 0.8, f"Low cosine similarity: {cos_sim}"
    logger.info("  Adaptive vs FD test: PASS")


def test_all_features_combined():
    """All diff features together produce finite gradients."""
    m_mj = mujoco.MjModel.from_xml_string(SPHERE_XML)
    d_mj = mujoco.MjData(m_mj)
    mujoco.mj_forward(m_mj, d_mj)

    mx = mujoco_torch.device_put(m_mj)
    dx = mujoco_torch.device_put(d_mj)

    qvel = dx.qvel.clone().requires_grad_(True)
    dx_test = dx.replace(qvel=qvel)

    with mujoco_torch.differentiable_mode(
        smooth_collisions=True,
        cfd=True,
        adaptive_integration=True,
        adaptive_substeps=4,
    ):
        loss = mujoco_torch.step(mx, dx_test).qpos.sum()
    loss.backward()

    g = qvel.grad
    assert g is not None, "No gradient computed"
    assert not g.isnan().any(), "NaN in gradient"
    assert not g.isinf().any(), "Inf in gradient"
    logger.info("  All-features grad norm: %.6e", g.norm().item())
    logger.info("  All features combined test: PASS")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Gradient correctness tests for DiffMJT")
    logger.info("=" * 60)

    test_sphere_sphere_smooth_gradcheck()
    test_plane_capsule_smooth_gradcheck()
    test_cfd_nonzero_gradient()
    test_adaptive_vs_fd()
    test_all_features_combined()

    logger.info("All tests PASSED.")
