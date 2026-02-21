#!/usr/bin/env python3
"""Visualize a mujoco-torch simulation using MuJoCo's passive viewer.

Demonstrates the device_get_into workflow: run physics in PyTorch via
mujoco-torch, then copy state back to a native MjData for rendering.

Run from the repo root:
    source .venv/bin/activate
    python examples/viewer_example.py        # Linux
    mjpython examples/viewer_example.py      # macOS (requires mjpython)
"""

import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from etils import epath

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()

NSTEPS = 10_000
SEED = 42

# ── Setup ────────────────────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)
d_mj = mujoco.MjData(m_mj)

# Small random velocity kick so the ant moves
d_mj.qvel[:] = 0.01 * np.random.randn(m_mj.nv)

mx = mujoco_torch.device_put(m_mj)
dx = mujoco_torch.device_put(d_mj)

# ── Simulate + Render ────────────────────────────────────────────────────────
with mujoco.viewer.launch_passive(m_mj, d_mj) as viewer:
    for _ in range(NSTEPS):
        # Read controls set via the viewer UI back into the torch state
        dx = dx.replace(ctrl=torch.as_tensor(d_mj.ctrl).clone())

        dx = mujoco_torch.step(mx, dx)
        mujoco_torch.device_get_into(d_mj, dx)
        viewer.sync()

        time.sleep(m_mj.opt.timestep)

        if not viewer.is_running():
            break

print(f"Ran {NSTEPS} steps.")
