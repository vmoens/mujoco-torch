"""Test torch.compile(fullgraph=True) compatibility for step().

Verifies that the entire step() function can be traced by Dynamo
without any graph breaks, for both compile-only and compile+vmap paths.
"""

import mujoco
import torch
from etils import epath

import mujoco_torch

torch.set_default_dtype(torch.float64)

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "humanoid.xml").read_text()

m_mj = mujoco.MjModel.from_xml_string(MODEL_XML)
d_mj = mujoco.MjData(m_mj)

mx = mujoco_torch.device_put(m_mj)
d = mujoco_torch.device_put(d_mj)

# Eager reference
d_eager = mujoco_torch.step(mx, d)
print("1. Eager: OK")

# vmap (no compile)
vmap_step = torch.vmap(lambda d: mujoco_torch.step(mx, d))
batch = torch.stack([d, d])
out = vmap_step(batch)
assert torch.allclose(out.qpos[0], d_eager.qpos, atol=1e-10)
print("2. vmap: OK")

# compile fullgraph=True (aot_eager backend — skips codegen)
compiled = torch.compile(
    lambda d: mujoco_torch.step(mx, d),
    fullgraph=True,
    backend="aot_eager",
)
out2 = compiled(d)
assert torch.allclose(out2.qpos, d_eager.qpos, atol=1e-10)
print("3. compile(fullgraph=True): OK")

# compile+vmap fullgraph=True
compiled_vmap = torch.compile(vmap_step, fullgraph=True, backend="aot_eager")
out3 = compiled_vmap(batch)
assert torch.allclose(out3.qpos[0], d_eager.qpos, atol=1e-10)
print("4. compile+vmap(fullgraph=True): OK")

print("\nAll modes pass with zero graph breaks!")
