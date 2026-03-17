"""Ant-v4 environment.

Observation: qpos[2:] (exclude x, y), qvel
Reward:      forward_velocity + healthy_reward - ctrl_cost
Termination: torso z outside [0.2, 1.0]

The bundled ``ant.xml`` is a dm_control-style fixed-base ant (no root joint).
We patch the XML at load time to insert a ``<freejoint/>`` on the torso, giving
the standard Gymnasium Ant-v4 locomotion semantics:

    nq = 15 (free: 7 + 8 hinge),  nv = 14 (free: 6 + 8 hinge),  nu = 8
"""

import re

import torch
from torchrl.data import Unbounded

from zoo.base import MujocoTorchEnv


class AntEnv(MujocoTorchEnv):
    """Ant: quadruped locomotion."""

    RESET_NOISE_SCALE = 0.1
    HEALTHY_Z_LOW = 0.2
    HEALTHY_Z_HIGH = 1.0
    HEALTHY_REWARD = 1.0
    CTRL_COST_WEIGHT = 0.5

    @classmethod
    def _xml_path(cls) -> str:
        return "ant.xml"

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        """Insert a freejoint on the torso so the ant can translate/rotate."""
        xml = super()._patch_xml(xml)
        return re.sub(
            r'(<body\s+name="torso"[^>]*>)',
            r"\1\n      <freejoint name='root'/>",
            xml,
            count=1,
        )

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        # nq=15 (free:7 + 8 hinge), nv=14 (free:6 + 8 hinge)
        # obs = qpos[2:] (13) + qvel (14) = 27
        return {
            "observation": Unbounded(
                shape=(num_envs, 27), dtype=dtype, device=device
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype)
        return {"observation": torch.cat([qpos[..., 2:], qvel], dim=-1)}

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
        z = self._dx.qpos[..., 2]
        return (z >= self.HEALTHY_Z_LOW) & (z <= self.HEALTHY_Z_HIGH)

    def _compute_terminated(self):
        return (~self._is_healthy()).unsqueeze(-1)
