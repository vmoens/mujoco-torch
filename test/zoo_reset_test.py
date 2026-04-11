from types import MethodType

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("mujoco")

import tensordict._unbatched as _ub

if not hasattr(_ub, "_HAS_WRAPPER_SUBCLASS_FIX"):
    _ub._HAS_WRAPPER_SUBCLASS_FIX = False

from mujoco_torch.zoo.base import MujocoTorchEnv


class FakeBatch:
    def __init__(self, qpos, qvel, ctrl):
        self.qpos = qpos
        self.qvel = qvel
        self.ctrl = ctrl

    def expand(self, n):
        return FakeBatch(
            self.qpos.expand(n, *self.qpos.shape[1:]),
            self.qvel.expand(n, *self.qvel.shape[1:]),
            self.ctrl.expand(n, *self.ctrl.shape[1:]),
        )

    def clone(self):
        return FakeBatch(self.qpos.clone(), self.qvel.clone(), self.ctrl.clone())

    def apply(self, fn, call_on_nested=False):
        return FakeBatch(fn(self.qpos), fn(self.qvel), fn(self.ctrl))

    def update_(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def __setitem__(self, index, other):
        self.qpos[index] = other.qpos
        self.qvel[index] = other.qvel
        self.ctrl[index] = other.ctrl


def _make_env(*, auto_reset):
    class DummyEnv:
        RESET_NOISE_SCALE = 0.0

    env = DummyEnv()
    env.dtype = torch.float32
    env.device = torch.device("cpu")
    env.num_envs = 4
    env.batch_size = torch.Size([4])
    env.auto_reset = auto_reset
    env.max_episode_steps = 5
    env._single_env = False
    env._physics_step = lambda data: data
    env._ctrl_dtype = torch.float64
    env._sim_dtype = torch.float64
    env._dx0 = FakeBatch(
        qpos=torch.zeros(1, 2, dtype=torch.float64),
        qvel=torch.zeros(1, 2, dtype=torch.float64),
        ctrl=torch.zeros(1, 1, dtype=torch.float64),
    )
    env._make_batch = MethodType(MujocoTorchEnv._make_batch, env)
    env._reset = MethodType(MujocoTorchEnv._reset, env)
    env._step = MethodType(MujocoTorchEnv._step, env)
    env._prepare_ctrl = MethodType(MujocoTorchEnv._prepare_ctrl, env)
    env._build_obs = lambda: {"observation": env._dx.qpos[..., :1].to(env.dtype)}
    env._compute_reward = (
        lambda qpos_before, action: torch.zeros(*env.batch_size, 1, dtype=env.dtype, device=env.device)
    )
    env._compute_terminated = (
        lambda: torch.zeros(*env.batch_size, 1, dtype=torch.bool, device=env.device)
    )
    return env


def test_partial_reset_preserves_float32_dtype():
    env = _make_env(auto_reset=False)
    out = env._reset()
    env._step_count[:] = 7

    env._reset(
        TensorDict(
            {"_reset": torch.tensor([[True], [False], [True], [False]], dtype=torch.bool)},
            batch_size=env.batch_size,
        )
    )

    assert out["observation"].dtype == torch.float32
    assert env._dx.qpos.dtype == torch.float64
    assert env._dx.qvel.dtype == torch.float64
    assert env._dx.ctrl.dtype == torch.float64
    assert torch.equal(env._step_count, torch.tensor([0, 7, 0, 7]))


def test_auto_reset_preserves_float32_dtype():
    env = _make_env(auto_reset=True)
    env._reset()
    env._step_count = torch.tensor([env.max_episode_steps - 1, 0, env.max_episode_steps - 1, 0], dtype=torch.long)

    out = env._step(
        TensorDict(
            {"action": torch.zeros(4, 1, dtype=env.dtype)},
            batch_size=env.batch_size,
        )
    )

    assert out["observation"].dtype == torch.float32
    assert env._dx.qpos.dtype == torch.float64
    assert env._dx.qvel.dtype == torch.float64
    assert env._dx.ctrl.dtype == torch.float64
    assert out["done"].squeeze(-1).tolist() == [True, False, True, False]
    assert torch.equal(env._step_count, torch.tensor([0, 1, 0, 1]))
