"""Humanoid-v4 environment.

Observation: qpos[2:] (exclude rootx, rooty), clipped qvel
Reward:      forward_velocity + healthy_reward - ctrl_cost
Termination: z < 1.0 or z > 2.0

nq = 28 (free: 7 + 21 hinge), nv = 27 (free: 6 + 21 hinge), nu = 21
"""

import torch
from torchrl.data import Unbounded

from mujoco_torch.zoo.base import MujocoTorchEnv, register_env


@register_env("humanoid")
class HumanoidEnv(MujocoTorchEnv):
    """Humanoid: bipedal locomotion."""

    RESET_NOISE_SCALE = 0.01
    FRAME_SKIP = 5
    HEALTHY_Z_LOW = 1.0
    HEALTHY_Z_HIGH = 2.0
    HEALTHY_REWARD = 5.0
    CTRL_COST_WEIGHT = 0.1

    @classmethod
    def _xml_path(cls) -> str:
        return "humanoid.xml"

    @classmethod
    def _camera_xml(cls) -> str:
        return '<camera name="side" pos="0 -6 3" xyaxes="1 0 0 0 0.45 1" fovy="60"/>'

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        # obs = qpos[2:] (26) + qvel (27) = 53
        return {
            "observation": Unbounded(shape=(num_envs, 53), dtype=dtype, device=device),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype).clamp(-10.0, 10.0)
        return {"observation": torch.cat([qpos[..., 2:], qvel], dim=-1)}

    def _compute_reward(self, qpos_before, action):
        forward_vel = (self._dx.qpos[..., 0] - qpos_before[..., 0]) / self._dt
        ctrl_cost = self.CTRL_COST_WEIGHT * (action**2).sum(dim=-1)
        healthy_reward = torch.where(self._is_healthy(), self.HEALTHY_REWARD, 0.0)
        reward = forward_vel + healthy_reward - ctrl_cost
        return reward.unsqueeze(-1).to(self.dtype)

    def _is_healthy(self):
        z = self._dx.qpos[..., 2]
        return (z >= self.HEALTHY_Z_LOW) & (z <= self.HEALTHY_Z_HIGH)

    def _compute_terminated(self):
        return (~self._is_healthy()).unsqueeze(-1)
