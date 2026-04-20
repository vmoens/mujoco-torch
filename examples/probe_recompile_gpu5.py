"""Minimal recompile probe mimicking bench.ipynb cell 1.

Compile vmap(step) on humanoid at B=128, call twice, check whether
torch._dynamo emits a recompile event on the second call.

Success = exactly ONE compile, no 'RECOMPILE' lines in TORCH_LOGS=recompiles.
"""

import os
import sys
import time

os.environ.setdefault("TORCH_LOGS", "recompiles")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import mujoco  # noqa: E402
import mujoco_torch  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402
from tensordict import UnbatchedTensor  # noqa: E402

from mujoco_torch._src import test_util  # noqa: E402

DEVICE = "cuda"
BATCH = 128
MODEL = "humanoid.xml"


def describe(tag, d):
    print(
        f"{tag}: nefc type={type(d.nefc).__name__} "
        f"ncon type={type(d.ncon).__name__} "
        f"batch_size={d.batch_size}"
    )
    if isinstance(d.nefc, UnbatchedTensor):
        print(f"       nefc.data shape={d.nefc.data.shape} stride={d.nefc.data.stride()}")
    else:
        print(f"       nefc shape={d.nefc.shape} stride={d.nefc.stride()}")


def make_batch():
    m_mj = test_util.load_test_file(MODEL)
    mx = mujoco_torch.device_put(m_mj)
    mx = mx.to(DEVICE)
    d_mj = mujoco.MjData(m_mj)
    with torch.device("cpu"):
        dx0 = mujoco_torch.device_put(d_mj)
    dx0 = dx0.to(DEVICE)
    return mx, dx0.expand(BATCH).clone()


def main():
    print(f"torch {torch.__version__}, cuda {torch.version.cuda}")
    print(f"gpu: {torch.cuda.get_device_name()}")

    torch._dynamo.reset()
    mx, dx = make_batch()
    describe("initial dx", dx)

    step = torch.compile(torch.vmap(mujoco_torch.step, in_dims=(None, 0)), fullgraph=True)

    for i in range(3):
        t0 = time.time()
        dx = step(mx, dx)
        torch.cuda.synchronize()
        dt = time.time() - t0
        describe(f"call {i + 1} (took {dt:.2f}s)", dx)

    print("\nif no 'Recompiling' lines above, the fix works.")


if __name__ == "__main__":
    main()
