"""Integration test: compile(vmap(step)) must not recompile on any env.

This test exists because Dynamo recompile regressions are expensive but hard
to catch — a recompile turns every subsequent call into a several-hundred-
second re-trace.  Running this under default `pytest` would block the suite
for 15+ minutes per env, so it is gated behind the ``integration`` marker.

Run with:  pytest -m integration test/compile_recompile_integration_test.py

See RELEASE.md for the pre-publish checklist.
"""

import pytest
import torch
import torch._dynamo
import torch._dynamo.utils

import mujoco_torch
from mujoco_torch._src import test_util

COMPILE_ENVS = [
    "ant.xml",
    "cartpole.xml",
    "halfcheetah.xml",
    "hopper.xml",
    "humanoid.xml",
    "swimmer.xml",
    "walker2d.xml",
]

BATCH = 4
NSTEPS = 3


def _unique_graph_count() -> int:
    return int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))


def _touch_all_fields(obj) -> None:
    """Recursively access every tensor in a tensorclass/tensordict so that
    any lazy type guard or re-wrap has a chance to fire."""

    def _poke(x):
        if isinstance(x, torch.Tensor):
            _ = x.shape
            _ = x.dtype
            _ = x.stride()

    torch.utils._pytree.tree_map(_poke, obj)


@pytest.mark.integration
@pytest.mark.parametrize("xml", COMPILE_ENVS)
def test_compile_vmap_step_no_recompile(xml):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float64)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    m = test_util.load_test_file(xml)
    with torch.device("cpu"):
        mx = mujoco_torch.device_put(m)
    mx = mx.to(device)

    with torch.device("cpu"):
        dx = mujoco_torch.make_data(mx)
    dx = dx.to(device).expand(BATCH).clone()

    compiled = torch.compile(torch.vmap(mujoco_torch.step, in_dims=(None, 0)), fullgraph=True)

    dx = compiled(mx, dx)
    if device == "cuda":
        torch.cuda.synchronize()
    _touch_all_fields(dx)
    graphs_after_first = _unique_graph_count()

    for _ in range(NSTEPS - 1):
        dx = compiled(mx, dx)
        if device == "cuda":
            torch.cuda.synchronize()
        _touch_all_fields(dx)

    graphs_after_all = _unique_graph_count()

    assert graphs_after_all == graphs_after_first, (
        f"Dynamo recompiled for {xml}: {graphs_after_first} unique graph(s) "
        f"after call 1, {graphs_after_all} after {NSTEPS} calls. "
        f"Set TORCH_LOGS=recompiles to see the triggering guard."
    )
