"""Shared helpers for benchmark tests."""

import mujoco
import numpy as np
import torch

import mujoco_torch
from mujoco_torch._src import test_util

NSTEPS = 20
SEED = 42


def load_model(name: str) -> mujoco.MjModel:
    return test_util.load_test_file(name)


def make_batch(mx, m_mj, batch_size, device="cpu"):
    """Create a batched Data with ``batch_size`` envs on ``device``."""
    rng = np.random.RandomState(SEED)
    qvels = 0.01 * rng.randn(batch_size, m_mj.nv)
    envs = []
    for i in range(batch_size):
        d = mujoco.MjData(m_mj)
        d.qvel[:] = qvels[i]
        envs.append(mujoco_torch.device_put(d))
    batched = torch.stack(envs, dim=0)
    if device != "cpu":
        batched = batched.to(device)
    return batched


def make_single(m_mj, device="cpu"):
    """Create a single Data on ``device``."""
    d = mujoco.MjData(m_mj)
    rng = np.random.RandomState(SEED)
    d.qvel[:] = 0.01 * rng.randn(m_mj.nv)
    dx = mujoco_torch.device_put(d)
    if device != "cpu":
        dx = dx.to(device)
    return dx
