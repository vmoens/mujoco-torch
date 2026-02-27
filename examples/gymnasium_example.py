#!/usr/bin/env python3
"""Gymnasium integration example for mujoco-torch.

Wraps mujoco-torch in a standard Gymnasium environment so it can be used
with any Gymnasium-compatible RL library.

Run:
    pip install gymnasium
    python examples/gymnasium_example.py
"""

import gymnasium as gym
import mujoco
import numpy as np
import torch
from etils import epath

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()


class MujocoTorchGymEnv(gym.Env):
    """Gymnasium wrapper around mujoco-torch."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, xml_string: str, max_episode_steps: int = 1000, render_mode=None, device=None):
        super().__init__()
        self.dtype = torch.float64
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        self._m_mj = mujoco.MjModel.from_xml_string(xml_string)
        self.mx = mujoco_torch.device_put(self._m_mj)
        self._device = device
        if device is not None:
            self.mx = self.mx.to(device)

        nq, nv, nu = self._m_mj.nq, self._m_mj.nv, self._m_mj.nu
        obs_size = nq + nv

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(nu,),
            dtype=np.float64,
        )

        self._dx = None
        self._step_count = 0
        self._prev_xpos = 0.0

        if render_mode == "human":
            self._d_mj = mujoco.MjData(self._m_mj)
            self._viewer = None

    def _get_obs(self):
        obs = torch.cat([self._dx.qpos, self._dx.qvel])
        return obs.detach().cpu().numpy()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        d_mj = mujoco.MjData(self._m_mj)
        self._dx = mujoco_torch.device_put(d_mj)
        if self._device is not None:
            self._dx = self._dx.to(self._device)
        self._step_count = 0
        self._prev_xpos = float(self._dx.xipos[1, 0])
        return self._get_obs(), {}

    def step(self, action):
        ctrl = torch.tensor(action, dtype=self.dtype)
        if self._device is not None:
            ctrl = ctrl.to(self._device)
        self._dx = self._dx.replace(ctrl=ctrl)
        self._dx = mujoco_torch.step(self.mx, self._dx)
        self._step_count += 1

        xpos = float(self._dx.xipos[1, 0])
        forward_reward = (xpos - self._prev_xpos) / self.mx.opt.timestep
        self._prev_xpos = xpos

        ctrl_cost = 0.5 * float((ctrl**2).sum())
        reward = forward_reward - ctrl_cost

        terminated = float(self._dx.qpos[2]) < 0.2
        truncated = self._step_count >= self.max_episode_steps

        obs = self._get_obs()
        info = {"x_position": xpos}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        mujoco_torch.device_get_into(self._d_mj, self._dx)
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._m_mj, self._d_mj)
        else:
            self._viewer.sync()

    def close(self):
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    env = MujocoTorchGymEnv(MODEL_XML, max_episode_steps=200)

    print("Running 100 random-action steps ...")
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    for _step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, info = env.reset()

    print(f"  Total reward over 100 steps: {total_reward:.2f}")
    print(f"  Final obs[:5]: {obs[:5]}")
    env.close()
