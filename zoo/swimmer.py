"""Swimmer-v4 environment.

Observation: qpos[2:] (skip free-joint x/y), qvel
Reward:      forward_velocity - ctrl_cost
Termination: never

Note: Swimmer uses a free joint, so qpos has 7 (quat) + 2 (hinge) = 9 dofs.
We skip the first two positional coords (x, y) from observations.
"""

import torch
from torchrl.data import Unbounded

from zoo.base import MujocoTorchEnv


class SwimmerEnv(MujocoTorchEnv):
    """Swimmer: multi-link aquatic locomotion."""

    CTRL_COST_WEIGHT = 1e-4

    @classmethod
    def _xml_path(cls) -> str:
        return "swimmer.xml"

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        # free joint: 7 qpos (3 pos + 4 quat) + 2 hinge = 9 total qpos
        # free joint: 6 qvel (3 lin + 3 ang) + 2 hinge = 8 total qvel
        # obs = qpos[2:] (7) + qvel (8) = 15
        # But swimmer typically skips the first 5 qpos (3pos + 2 of quat?)
        # Actually gymnasium skips first 2 qpos for swimmer (x, y position)
        # qpos[2:] = 7, qvel = 8 => 15
        return {
            "observation": Unbounded(shape=(num_envs, 13), dtype=dtype, device=device),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype)
        # Skip x, y, z position but keep quaternion and hinge angles
        # For a 2D swimmer with free joint: qpos = [x,y,z, qw,qx,qy,qz, j1, j2]
        # Gymnasium-v4 skips first 2 (x, y)
        # qvel for free joint = [vx,vy,vz, wx,wy,wz, dj1, dj2]
        # Gymnasium-v4 skips first 2 velocities (vx, vy)
        obs = torch.cat([qpos[..., 2:], qvel[..., 2:]], dim=-1)
        return {"observation": obs}

    def _compute_reward(self, qpos_before, action):
        forward_vel = (self._dx.qpos[..., 0] - qpos_before[..., 0]) / self._dt
        ctrl_cost = self.CTRL_COST_WEIGHT * (action**2).sum(dim=-1)
        reward = forward_vel - ctrl_cost
        return reward.unsqueeze(-1).to(self.dtype)

    def _compute_terminated(self):
        return torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
