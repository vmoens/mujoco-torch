"""Tests for the CMG cluster math and the random unit quaternion sampler."""

import math

import torch

from mujoco_torch import random_unit_quat
from mujoco_torch.zoo.cmg import (
    cmg_jacobian,
    manipulability,
    orthogonal_6cmg_geometry,
    pyramid_4cmg_geometry,
    rodrigues_rotate,
)


def test_random_unit_quat_shape_and_norm():
    q = random_unit_quat((5, 3))
    assert q.shape == torch.Size([5, 3, 4])
    norms = q.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_random_unit_quat_generator_is_deterministic():
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    a = random_unit_quat((4,), generator=g1)
    b = random_unit_quat((4,), generator=g2)
    torch.testing.assert_close(a, b)


def test_random_unit_quat_dtype_and_device():
    q = random_unit_quat((2,), dtype=torch.float64)
    assert q.dtype is torch.float64


def test_rodrigues_rotate_identity_angle():
    """Rotating by zero leaves the vector unchanged."""
    axis = torch.tensor([[1.0], [0.0], [0.0]])
    vec = torch.tensor([[0.0], [1.0], [0.0]])
    out = rodrigues_rotate(axis, vec, torch.zeros(1))
    torch.testing.assert_close(out.squeeze(-1).squeeze(-1), vec.squeeze(-1))


def test_rodrigues_rotate_quarter_turn():
    """Quarter turn around +z maps +x to +y."""
    axis = torch.tensor([[0.0], [0.0], [1.0]])
    vec = torch.tensor([[1.0], [0.0], [0.0]])
    angle = torch.tensor([math.pi / 2.0])
    out = rodrigues_rotate(axis, vec, angle).squeeze(-1).squeeze(-1)
    torch.testing.assert_close(out, torch.tensor([0.0, 1.0, 0.0]), atol=1e-6, rtol=0)


def test_pyramid_4cmg_geometry_shapes_and_orthogonality():
    g, r0 = pyramid_4cmg_geometry()
    assert g.shape == torch.Size([3, 4])
    assert r0.shape == torch.Size([3, 4])
    # Each rotor reference must be orthogonal to its gimbal axis.
    dot = (g * r0).sum(0)
    torch.testing.assert_close(dot, torch.zeros(4), atol=1e-6, rtol=0)
    # Gimbal axes have unit norm (within rounding of the math).
    torch.testing.assert_close(g.norm(dim=0), torch.ones(4), atol=1e-6, rtol=0)


def test_orthogonal_6cmg_geometry_shapes_and_orthogonality():
    g, r0 = orthogonal_6cmg_geometry()
    assert g.shape == torch.Size([3, 6])
    assert r0.shape == torch.Size([3, 6])
    dot = (g * r0).sum(0)
    torch.testing.assert_close(dot, torch.zeros(6), atol=1e-6, rtol=0)


def test_cmg_jacobian_shape_unbatched():
    g, r0 = pyramid_4cmg_geometry()
    j = cmg_jacobian(torch.zeros(4), g, r0, h=1.0)
    assert j.shape == torch.Size([3, 4])


def test_cmg_jacobian_shape_batched():
    g, r0 = pyramid_4cmg_geometry()
    j = cmg_jacobian(torch.zeros(7, 4), g, r0, h=1.0)
    assert j.shape == torch.Size([7, 3, 4])


def test_cmg_jacobian_scales_linearly_with_h():
    g, r0 = pyramid_4cmg_geometry()
    j1 = cmg_jacobian(torch.zeros(4), g, r0, h=1.0)
    j3 = cmg_jacobian(torch.zeros(4), g, r0, h=3.0)
    torch.testing.assert_close(j3, 3.0 * j1)


def test_manipulability_pyramid_and_orthogonal_neutral():
    """At the neutral configuration, both standard clusters are full-rank."""
    g4, r4 = pyramid_4cmg_geometry()
    j4 = cmg_jacobian(torch.zeros(4), g4, r4, h=1.0)
    m4 = manipulability(j4)
    assert m4.item() > 0.5

    g6, r6 = orthogonal_6cmg_geometry()
    j6 = cmg_jacobian(torch.zeros(6), g6, r6, h=1.0)
    m6 = manipulability(j6)
    assert m6.item() > 0.5


def test_manipulability_singular_jacobian():
    """A rank-1 Jacobian has near-zero ``det(J J^T)``; the ``+eps`` floor
    keeps :func:`manipulability` finite at the singularity."""
    rank1 = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    m = manipulability(rank1, eps=1e-8)
    assert m.item() < 1e-3
    assert torch.isfinite(m).all()


def test_manipulability_batched():
    g, r0 = pyramid_4cmg_geometry()
    j = cmg_jacobian(torch.zeros(5, 4), g, r0, h=1.0)
    m = manipulability(j)
    assert m.shape == torch.Size([5])
    assert (m > 0).all()
