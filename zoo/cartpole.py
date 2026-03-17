"""CartPole (Inverted Pendulum) environment.

Observation: qpos, qvel
Reward:      1.0 per step (survive)
Termination: |pole angle| > 0.2
"""

import torch
from torchrl.data import Unbounded

from zoo.base import MujocoTorchEnv


class CartPoleEnv(MujocoTorchEnv):
    """CartPole / Inverted Pendulum: balance a pole on a cart."""

    ANGLE_LIMIT = 0.2

    @classmethod
    def _xml_path(cls) -> str:
        return "cartpole.xml"

    @classmethod
    def _camera_xml(cls) -> str:
        return (
            '<camera name="side" pos="0 -2 1.5" '
            'xyaxes="1 0 0 0 0.45 1" fovy="60"/>'
        )

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        # nq=2 (slider + hinge), nv=2 => obs dim = 4
        return {
            "observation": Unbounded(
                shape=(num_envs, 4), dtype=dtype, device=device
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype)
        return {"observation": torch.cat([qpos, qvel], dim=-1)}

    def _compute_reward(self, qpos_before, action):
        return torch.ones(
            *self.batch_size, 1, dtype=self.dtype, device=self.device
        )

    def _compute_terminated(self):
        angle = self._dx.qpos[..., 1]  # hinge joint = pole angle
        return (angle.abs() > self.ANGLE_LIMIT).unsqueeze(-1)
