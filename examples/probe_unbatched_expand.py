"""Probe: does UnbatchedTensor survive .expand(B).clone() through TensorClass?

Run on CPU first. If the wrapping is preserved, we can use UnbatchedTensor for
nefc/ncon/Contact broadcast fields and avoid the stride-0 recompile under vmap.
"""

import mujoco_torch  # noqa: F401 — apply patches
import torch
from tensordict import UnbatchedTensor

from mujoco_torch._src.types import Data
from mujoco_torch._src import test_util

torch.manual_seed(0)


def is_unbatched(x) -> bool:
    return isinstance(x, UnbatchedTensor)


def describe(tag: str, x):
    stride = None
    try:
        stride = x.stride()
    except AttributeError:
        pass
    print(f"  {tag}: type={type(x).__name__} shape={tuple(x.shape)} stride={stride} dtype={x.dtype}")


m = test_util.load_test_file("humanoid.xml")
d = mujoco_torch.make_data(mujoco_torch.device_put(m))
print("make_data (no wrap):")
describe("nefc", d.nefc)
describe("ncon", d.ncon)
print(f"  is_unbatched(nefc)={is_unbatched(d.nefc)}")

# Manually wrap nefc as UnbatchedTensor, leave everything else alone.
wrapped_nefc = UnbatchedTensor(d.nefc.clone())
wrapped_ncon = UnbatchedTensor(d.ncon.clone())
d2 = d.replace(nefc=wrapped_nefc, ncon=wrapped_ncon)
print("\nafter replace with UnbatchedTensor:")
describe("nefc", d2.nefc)
describe("ncon", d2.ncon)
print(f"  is_unbatched(nefc)={is_unbatched(d2.nefc)}")

B = 4
d3 = d2.expand(B).clone()
print(f"\nafter .expand({B}).clone():")
describe("nefc", d3.nefc)
describe("ncon", d3.ncon)
print(f"  is_unbatched(nefc)={is_unbatched(d3.nefc)}")
print(f"  is_unbatched(ncon)={is_unbatched(d3.ncon)}")

print("\nqpos (sanity, should be regular batched tensor):")
describe("qpos", d3.qpos)
print(f"  is_unbatched(qpos)={is_unbatched(d3.qpos)}")

# --- Key test: full vmap+compile pipeline, check no recompile on call 2. ---
import os
os.environ.setdefault("TORCH_LOGS", "recompiles")
import torch._dynamo

mx = mujoco_torch.device_put(m)

def make_batch(B):
    d = mujoco_torch.make_data(mx)
    d = d.replace(
        nefc=UnbatchedTensor(d.nefc.clone()),
        ncon=UnbatchedTensor(d.ncon.clone()),
    )
    return d.expand(B).clone()

torch._dynamo.reset()

step = torch.compile(torch.vmap(mujoco_torch.step, in_dims=(None, 0)), fullgraph=True)

B = 4
print(f"\n=== compile+vmap+step, B={B} ===")
for i in range(3):
    d_in = make_batch(B)
    print(f"call {i + 1}: input nefc type={type(d_in.nefc).__name__} ncon type={type(d_in.ncon).__name__}")
    d_out = step(mx, d_in)
    print(f"        output nefc type={type(d_out.nefc).__name__} ncon type={type(d_out.ncon).__name__}")

