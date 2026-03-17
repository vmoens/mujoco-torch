"""Walker2d-v4 environment.

Observation: qpos[1:] (exclude rootx), clipped qvel
Reward:      forward_velocity + healthy_reward - ctrl_cost
Termination: z outside [0.8, 2.0] or |angle| > 1.0
"""

import torch
from torchrl.data import Unbounded

from zoo.base import MujocoTorchEnv


class Walker2dEnv(MujocoTorchEnv):
    """Walker2d: bipedal locomotion."""

    HEALTHY_Z_LOW = 0.8
    HEALTHY_Z_HIGH = 2.0
    HEALTHY_ANGLE_MAX = 1.0
    HEALTHY_REWARD = 1.0
    CTRL_COST_WEIGHT = 1e-3

    @classmethod
    def _xml_path(cls) -> str:
        return "walker2d.xml"

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        # nq=9 (rootx, rootz, rooty + 6 hinge), nv=9
        # obs = qpos[1:] (8) + clipped qvel (9) = 17
        return {
            "observation": Unbounded(
                shape=(num_envs, 17), dtype=dtype, device=device
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype).clamp(-10.0, 10.0)
        return {"observation": torch.cat([qpos[..., 1:], qvel], dim=-1)}

    def _compute_reward(self, qpos_before, action):
        forward_vel = (
            (self._dx.qpos[..., 0] - qpos_before[..., 0]) / self._dt
        )
        ctrl_cost = self.CTRL_COST_WEIGHT * (action**2).sum(dim=-1)
        healthy_reward = torch.where(
            self._is_healthy(), self.HEALTHY_REWARD, 0.0
        )
        reward = forward_vel + healthy_reward - ctrl_cost
        return reward.unsqueeze(-1).to(self.dtype)

    def _is_healthy(self):
        z = self._dx.qpos[..., 1]  # rootz
        angle = self._dx.qpos[..., 2]  # rooty
        return (
            (z >= self.HEALTHY_Z_LOW)
            & (z <= self.HEALTHY_Z_HIGH)
            & (angle.abs() <= self.HEALTHY_ANGLE_MAX)
        )

    def _compute_terminated(self):
        return (~self._is_healthy()).unsqueeze(-1)
