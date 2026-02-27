#!/usr/bin/env python3
"""TorchRL integration example for mujoco-torch.

Demonstrates wrapping mujoco-torch in a TorchRL EnvBase so it can be used
with TorchRL's data collectors, replay buffers, and policy trainers.

Run:
    pip install torchrl
    python examples/torchrl_example.py
"""

import mujoco
import torch
from etils import epath
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()


class MujocoTorchEnv(EnvBase):
    """A TorchRL environment backed by mujoco-torch."""

    def __init__(self, xml_string: str, max_episode_steps: int = 1000, device=None, dtype=torch.float64):
        super().__init__(device=device)
        self.dtype = dtype
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        self._m_mj = m_mj
        self.mx = mujoco_torch.device_put(m_mj)
        if device is not None:
            self.mx = self.mx.to(device)

        self.nq = m_mj.nq
        self.nv = m_mj.nv
        self.nu = m_mj.nu

        from torchrl.data import Bounded, Composite, Unbounded

        self.observation_spec = Composite(
            qpos=Unbounded(shape=(self.nq,), dtype=self.dtype, device=self.device),
            qvel=Unbounded(shape=(self.nv,), dtype=self.dtype, device=self.device),
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=self.dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(1,), dtype=self.dtype, device=self.device)

    def _reset(self, tensordict=None, **kwargs):
        d_mj = mujoco.MjData(self._m_mj)
        self._dx = mujoco_torch.device_put(d_mj)
        if self.device is not None:
            self._dx = self._dx.to(self.device)
        self._step_count = 0
        self._prev_xpos = self._dx.xipos[1, 0].clone()

        return TensorDict(
            {
                "qpos": self._dx.qpos.to(self.dtype),
                "qvel": self._dx.qvel.to(self.dtype),
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            batch_size=[],
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"].to(self.dtype)
        self._dx = self._dx.replace(ctrl=action)
        self._dx = mujoco_torch.step(self.mx, self._dx)
        self._step_count += 1

        xpos = self._dx.xipos[1, 0]
        forward_reward = (xpos - self._prev_xpos) / self.mx.opt.timestep
        self._prev_xpos = xpos.clone()

        ctrl_cost = 0.5 * (action**2).sum()
        reward = (forward_reward - ctrl_cost).unsqueeze(0).to(self.dtype)

        terminated = self._dx.qpos[2] < 0.2
        truncated = self._step_count >= self.max_episode_steps
        done = terminated | truncated

        return TensorDict(
            {
                "qpos": self._dx.qpos.to(self.dtype),
                "qvel": self._dx.qvel.to(self.dtype),
                "reward": reward,
                "done": done.unsqueeze(0),
                "terminated": terminated.unsqueeze(0),
            },
            batch_size=[],
            device=self.device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    env = MujocoTorchEnv(MODEL_XML, max_episode_steps=200)
    print("Checking env specs ...")
    check_env_specs(env, seed=42)
    print("Env specs OK!")

    print("\nRunning 100 random-action steps ...")
    td = env.reset()
    total_reward = 0.0
    for _step in range(100):
        action = env.action_spec.rand()
        td["action"] = action
        td = env.step(td)["next"]
        total_reward += td["reward"].item()

    print(f"  Total reward over 100 steps: {total_reward:.2f}")
    print(f"  Final qpos[:3]: {td['qpos'][:3].tolist()}")
