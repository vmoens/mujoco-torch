"""Dump torch.compile output_code for a compiled mujoco-torch step.

Usage:
    TORCH_LOGS="output_code" python scripts/dump_output_code.py 2>&1 | tee /tmp/output_code.txt
"""

import mujoco
import numpy as np
import torch

import mujoco_torch
from mujoco_torch._src import test_util

BATCH_SIZE = 64


def main():
    m_mj = test_util.load_test_file("ant.xml")

    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m_mj)
    mx = mx.to("cuda")

    # Warm caches with a single non-vmapped step
    d_mj = mujoco.MjData(m_mj)
    rng = np.random.RandomState(42)
    d_mj.qvel[:] = 0.01 * rng.randn(m_mj.nv)
    with torch.device("cpu"):
        dx_warm = mujoco_torch.device_put(d_mj)
    dx_warm = dx_warm.to("cuda")
    mujoco_torch.step(mx, dx_warm)
    torch.cuda.synchronize()

    # Build a batch
    envs = []
    qvels = 0.01 * rng.randn(BATCH_SIZE, m_mj.nv)
    with torch.device("cpu"):
        for i in range(BATCH_SIZE):
            d = mujoco.MjData(m_mj)
            d.qvel[:] = qvels[i]
            envs.append(mujoco_torch.device_put(d))
    d_batch = torch.stack([e.to("cuda") for e in envs], dim=0)

    # Compile
    vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
    compiled_fn = torch.compile(vmap_step, fullgraph=True)

    # Warmup (triggers compilation + output_code logging)
    for _ in range(3):
        d_batch = compiled_fn(d_batch)
    torch.cuda.synchronize()

    print("=== Done. Output code was logged above. ===")


if __name__ == "__main__":
    main()
