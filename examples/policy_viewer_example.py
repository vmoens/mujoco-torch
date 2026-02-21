#!/usr/bin/env python3
"""Neural-network policy driving an ant in the MuJoCo viewer.

A small torch.nn MLP maps (qpos, qvel) to actuator controls at each
timestep — no JAX, no framework glue, just native PyTorch end-to-end.

Pass --compile to fuse the policy + physics step into a single compiled
function via torch.compile, showing that mujoco-torch composes with the
standard PyTorch compiler stack.

The policy starts with random weights so the ant will flail.  The point
is to show that a standard torch.nn.Module plugs directly into the
mujoco-torch simulation loop.

Run from the repo root:
    source .venv/bin/activate
    python examples/policy_viewer_example.py              # Linux
    python examples/policy_viewer_example.py --compile    # Linux, compiled
    mjpython examples/policy_viewer_example.py            # macOS (requires mjpython)
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
from etils import epath
from torch import nn

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()

NSTEPS = 10_000
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Policy-driven ant viewer")
    parser.add_argument("--compile", action="store_true", help="torch.compile the policy + step")
    return parser.parse_args()


def main(args):
    # ── Policy ───────────────────────────────────────────────────────────
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_default_dtype(torch.float64)

    m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)

    policy = nn.Sequential(
        nn.Linear(m_mj.nq + m_mj.nv, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, m_mj.nu),
        nn.Tanh(),
    )

    # ── Setup ────────────────────────────────────────────────────────────
    d_mj = mujoco.MjData(m_mj)
    d_mj.qvel[:] = 0.01 * np.random.randn(m_mj.nv)

    mx = mujoco_torch.device_put(m_mj)
    dx = mujoco_torch.device_put(d_mj)

    @torch.no_grad()
    def policy_step(mx, dx):
        obs = torch.cat([dx.qpos, dx.qvel])
        ctrl = policy(obs)
        dx = dx.replace(ctrl=ctrl)
        return mujoco_torch.step(mx, dx)

    step_fn = policy_step
    if args.compile:
        step_fn = torch.compile(policy_step)

    # ── Simulate + Render ────────────────────────────────────────────────
    if args.compile:
        print("Compiling policy + step ...")
        t0 = time.perf_counter()
        step_fn(mx, dx)
        dx = mujoco_torch.device_put(d_mj)
        print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    with mujoco.viewer.launch_passive(m_mj, d_mj) as viewer:
        for _ in range(NSTEPS):
            dx = step_fn(mx, dx)
            mujoco_torch.device_get_into(d_mj, dx)
            viewer.sync()

            time.sleep(m_mj.opt.timestep)

            if not viewer.is_running():
                break

    print(f"Ran {NSTEPS} steps.")


if __name__ == "__main__":
    main(parse_args())
