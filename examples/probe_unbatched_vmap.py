"""Probe 2: does vmap preserve UnbatchedTensor in Data fields (no compile)."""

import mujoco_torch  # noqa: F401
import torch
from tensordict import UnbatchedTensor

from mujoco_torch._src import test_util


def summary(tag, d):
    print(f"{tag}: nefc type={type(d.nefc).__name__}  ncon type={type(d.ncon).__name__}")
    print(f"       nefc.data={d.nefc.data if isinstance(d.nefc, UnbatchedTensor) else d.nefc}")


m_mj = test_util.load_test_file("humanoid.xml")
mx = mujoco_torch.device_put(m_mj)

d = mujoco_torch.make_data(mx)
summary("make_data", d)

B = 4
d_batch = d.expand(B).clone()
summary(f"expand({B}).clone", d_batch)

print("\nrun vmap(step) once (no compile):")
out = torch.vmap(mujoco_torch.step, in_dims=(None, 0))(mx, d_batch)
summary("vmap(step) out", out)

# Call #2 — is the output's nefc/ncon the same type/stride?
out2 = torch.vmap(mujoco_torch.step, in_dims=(None, 0))(mx, out)
summary("vmap(step) out2", out2)
