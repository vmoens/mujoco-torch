"""Shared helpers for benchmark tests."""

import mujoco
import numpy as np
import torch

import mujoco_torch
from mujoco_torch._src import test_util

NSTEPS = 100
SEED = 42
ROUNDS = 2
WARMUP_ROUNDS = 1
COMPILE_WARMUP_ITERS = 5
DEVICE = "cuda"


def load_model(name: str) -> mujoco.MjModel:
    """Load a MuJoCo model by name (e.g. ``'humanoid'`` or ``'humanoid.xml'``)."""
    if not name.endswith(".xml"):
        name = f"{name}.xml"
    return test_util.load_test_file(name)


def make_batch(mx, m_mj, batch_size, device=DEVICE):
    """Create a batched Data with ``batch_size`` envs on ``device``."""
    rng = np.random.RandomState(SEED)
    qvels = 0.01 * rng.randn(batch_size, m_mj.nv)
    envs = []
    with torch.device("cpu"):
        for i in range(batch_size):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d))
    return torch.stack([e.to(device) for e in envs], dim=0)


def make_single(m_mj, device=DEVICE):
    """Create a single Data on ``device``."""
    d = mujoco.MjData(m_mj)
    rng = np.random.RandomState(SEED)
    d.qvel[:] = 0.01 * rng.randn(m_mj.nv)
    with torch.device("cpu"):
        dx = mujoco_torch.device_put(d)
    if device != "cpu":
        dx = dx.to(device)
    return dx


def warm_caches(mx, m_mj, device=DEVICE):
    """Single non-vmapped step to populate _CachedConst caches on the target device."""
    d_mj = mujoco.MjData(m_mj)
    rng = np.random.RandomState(SEED)
    d_mj.qvel[:] = 0.01 * rng.randn(m_mj.nv)
    with torch.device("cpu"):
        dx = mujoco_torch.device_put(d_mj)
    dx = dx.to(device)
    mujoco_torch.step(mx, dx)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
