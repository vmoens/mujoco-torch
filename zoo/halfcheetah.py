"""HalfCheetah-v4 environment.

Observation: qpos[1:] (exclude rootx), qvel
Reward:      forward_velocity - 0.1 * ctrl_cost
Termination: never
"""

import torch
from torchrl.data import Unbounded

from zoo.base import MujocoTorchEnv


class HalfCheetahEnv(MujocoTorchEnv):
    """HalfCheetah: run forward as fast as possible."""

    @classmethod
    def _xml_path(cls) -> str:
        return "halfcheetah.xml"

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        return {
            "observation": Unbounded(
                shape=(num_envs, 17), dtype=dtype, device=device
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype)
        return {"observation": torch.cat([qpos[..., 1:], qvel], dim=-1)}

    def _compute_reward(self, qpos_before, action):
        forward_vel = (
            (self._dx.qpos[..., 0] - qpos_before[..., 0]) / self._dt
        )
        ctrl_cost = (action**2).sum(dim=-1)
        return (forward_vel - 0.1 * ctrl_cost).unsqueeze(-1).to(self.dtype)

    def _compute_terminated(self):
        return torch.zeros(
            *self.batch_size, 1, dtype=torch.bool, device=self.device
        )
