"""``torch.compile(fullgraph=True)`` over batched ``vmap(step)`` on CPU.

These checks trace the whole batched physics step through Dynamo and
AOTAutograd (``backend="aot_eager"``) so that graph breaks, recompiles, and
tracing failures surface as loud test errors in a few tens of seconds.  The
Inductor variants are gated behind the ``integration`` marker because C++
code generation for a full step takes minutes on CPU.
"""

import pytest
import torch
import torch._dynamo
import torch._dynamo.utils

import mujoco_torch
from mujoco_torch._src import test_util

BATCH = 2
NSTEPS = 3


def _unique_graph_count() -> int:
    return int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))


def _reset_dynamo():
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()


def _hopper_batch():
    m = test_util.load_test_file("hopper.xml")
    mx = mujoco_torch.device_put(m)
    dx = mujoco_torch.make_data(mx).expand(BATCH).clone()
    dx.qvel[:] = 0.01 * torch.randn(BATCH, mx.nv, generator=torch.Generator().manual_seed(0), dtype=dx.qvel.dtype)
    return mx, dx


def _assert_data_close(actual, expected):
    for name in ("time", "qpos", "qvel", "qacc", "sensordata", "efc_force"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), msg=name)


def _run_compiled_vs_eager_step(backend):
    _reset_dynamo()
    mx, dx = _hopper_batch()
    vmap_step = torch.vmap(mujoco_torch.step, in_dims=(None, 0))
    compiled = torch.compile(vmap_step, backend=backend, fullgraph=True)

    dx_eager = dx_compiled = dx
    for _ in range(NSTEPS):
        dx_eager = vmap_step(mx, dx_eager)
        dx_compiled = compiled(mx, dx_compiled)
        _assert_data_close(dx_compiled, dx_eager)

    assert _unique_graph_count() == 1, (
        f"compile(vmap(step), fullgraph=True) produced {_unique_graph_count()} graphs "
        f"over {NSTEPS} calls; expected a single graph with no recompiles."
    )


def test_compile_vmap_step_fullgraph_aot_eager():
    _run_compiled_vs_eager_step("aot_eager")


@pytest.mark.integration
def test_compile_vmap_step_fullgraph_inductor():
    _run_compiled_vs_eager_step("inductor")


def _zoo_trajectories(compile_kwargs, nsteps=5):
    from mujoco_torch.zoo import HopperEnv

    torch.manual_seed(0)
    env = HopperEnv(num_envs=BATCH, compile_step=compile_kwargs is not None, compile_kwargs=compile_kwargs)
    td = env.reset()
    nu = env.action_spec.shape[-1]
    obs, rewards = [td["observation"]], []
    for i in range(nsteps):
        # Deterministic, non-trivial actions shared by both runs.
        action = torch.sin(torch.arange(BATCH * nu, dtype=env.dtype).view(BATCH, nu) + i)
        td = env.step(td.update({"action": action}))
        obs.append(td["next", "observation"])
        rewards.append(td["next", "reward"])
        td = td["next"]
    return torch.stack(obs), torch.stack(rewards)


def _run_zoo_compiled_vs_eager(compile_kwargs):
    pytest.importorskip("torchrl")
    _reset_dynamo()
    obs_eager, rew_eager = _zoo_trajectories(None)
    obs_compiled, rew_compiled = _zoo_trajectories(compile_kwargs)
    torch.testing.assert_close(obs_compiled, obs_eager)
    torch.testing.assert_close(rew_compiled, rew_eager)
    assert _unique_graph_count() == 1, (
        f"the zoo physics step compiled into {_unique_graph_count()} graphs; expected exactly one."
    )


def test_zoo_compiled_step_matches_eager_aot_eager():
    _run_zoo_compiled_vs_eager({"backend": "aot_eager"})


@pytest.mark.integration
def test_zoo_compiled_step_matches_eager_inductor():
    _run_zoo_compiled_vs_eager({})


@pytest.mark.integration
def test_zoo_compiled_rollout():
    """The batched compiled Hopper rollout from the issue report must complete."""
    pytest.importorskip("torchrl")
    from mujoco_torch.zoo import HopperEnv

    _reset_dynamo()
    env = HopperEnv(num_envs=BATCH, compile_step=True)
    # Hopper terminates within a few steps under random actions, so the
    # rollout may legitimately stop early; what matters is that the compiled
    # step ran and produced finite rewards.
    td = env.rollout(3)
    assert td.shape[0] == BATCH and 1 <= td.shape[1] <= 3
    assert torch.isfinite(td["next", "reward"]).all()
    assert _unique_graph_count() == 1
