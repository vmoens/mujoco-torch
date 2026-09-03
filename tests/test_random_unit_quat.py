"""Tests for the random unit quaternion sampler."""

import torch

from mujoco_torch import random_unit_quat


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
