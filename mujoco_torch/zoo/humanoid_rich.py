"""Humanoid with rich observations matching Gymnasium Humanoid-v5.

Adds cinert, cvel, and qfrc_actuator to the observation space,
giving the policy body inertia, COM velocity, and actuator force
information that is critical for learning bipedal locomotion.

Observation: qpos[2:] (26) + qvel_clipped (27) + cinert[1:].flat (160)
             + cvel[1:].flat (96) + qfrc_actuator (27) = 336

Reward:      forward_velocity + healthy_reward - ctrl_cost
Termination: z < 1.0 or z > 2.0

nq = 28 (free: 7 + 21 hinge), nv = 27 (free: 6 + 21 hinge), nu = 21
nbody = 17 (including world)
"""

import torch
from torchrl.data import Unbounded

from mujoco_torch.zoo.base import register_env
from mujoco_torch.zoo.humanoid import HumanoidEnv


# Dimensions: nbody=17, skip world body → 16 bodies
# cinert: 16 * 10 = 160
# cvel:   16 * 6  = 96
# qfrc_actuator: nv = 27
_RICH_OBS_DIM = 26 + 27 + 160 + 96 + 27  # = 336


@register_env("humanoid_rich")
class HumanoidRichEnv(HumanoidEnv):
    """Humanoid with rich observations (cinert, cvel, qfrc_actuator)."""

    @staticmethod
    def _obs_spec_dict(num_envs, dtype, device):
        return {
            "observation": Unbounded(
                shape=(num_envs, _RICH_OBS_DIM), dtype=dtype, device=device,
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype).clamp(-10.0, 10.0)

        # Body-level quantities (skip world body at index 0)
        cinert = self._dx.cinert[..., 1:, :].to(self.dtype).flatten(-2)
        cvel = self._dx.cvel[..., 1:, :].to(self.dtype).flatten(-2)
        qfrc_actuator = self._dx.qfrc_actuator.to(self.dtype)

        return {
            "observation": torch.cat(
                [qpos[..., 2:], qvel, cinert, cvel, qfrc_actuator], dim=-1,
            ),
        }
