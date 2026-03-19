"""Satellite CMG environments.

Two satellite variants with Control Moment Gyros (CMGs) for attitude control.

``SatelliteLargeEnv``
    High-altitude satellite with 4 CMGs in a pyramid arrangement.
    nq=15, nv=14, nu=8 (agent controls 4 gimbal rates; 4 rotor speeds fixed).
    obs_dim = 23: bus_quat(4) + bus_omega(3) + cmg_angles(8) + cmg_rates(8).

``SatelliteSmallEnv``
    Low-altitude CubeSat with 6 CMGs for redundant attitude control.
    nq=19, nv=18, nu=12 (agent controls 6 gimbal rates; 6 rotor speeds fixed).
    obs_dim = 31: bus_quat(4) + bus_omega(3) + cmg_angles(12) + cmg_rates(12).

Reward
    sun_alignment - ctrl_cost - angular_velocity_penalty

    The sun is at +Z in the world frame.  ``sun_alignment`` is the
    dot product of the body's +Z axis with the sun direction (-1..1).

Termination
    Never (satellites don't crash; episodes end by truncation only).
"""

import re

import torch
from torchrl.data import Bounded, Unbounded

from mujoco_torch.zoo.base import MujocoTorchEnv, register_env


class _SatelliteBase(MujocoTorchEnv):
    """Common base for satellite CMG environments."""

    N_GIMBALS: int
    ROTOR_SPEED: float = 100.0
    FRAME_SKIP = 10
    RESET_NOISE_SCALE = 0.001
    CTRL_COST_WEIGHT = 0.01
    ANG_VEL_WEIGHT = 0.1
    RENDER_BACKGROUND = (0.0, 0.0, 0.05)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, self.N_GIMBALS),
            dtype=self.dtype,
            device=self.device,
        )

    @classmethod
    def _patch_xml(cls, xml: str) -> str:
        """Replace cameras/lights but skip floor (space has no ground)."""
        xml = re.sub(r"<camera\b[^/]*/>\s*", "", xml)
        xml = re.sub(r"<light\b[^/]*/>\s*", "", xml)
        camera = cls._camera_xml()
        light = (
            '<light name="top" pos="0 0 4" dir="0 0 -1" '
            'diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3" '
            'directional="true"/>'
        )
        xml = xml.replace(
            "<worldbody>",
            f"<worldbody>\n  {camera}\n  {light}",
        )
        return xml

    @classmethod
    def _obs_spec_dict(cls, num_envs, dtype, device):
        obs_dim = 7 + 4 * cls.N_GIMBALS
        return {
            "observation": Unbounded(
                shape=(num_envs, obs_dim),
                dtype=dtype,
                device=device,
            ),
        }

    def _make_obs(self):
        qpos = self._dx.qpos.to(self.dtype)
        qvel = self._dx.qvel.to(self.dtype)
        return {
            "observation": torch.cat(
                [
                    qpos[..., 3:7],  # bus quaternion (w, x, y, z)
                    qvel[..., 3:6],  # bus angular velocity
                    qpos[..., 7:],  # CMG joint angles
                    qvel[..., 6:],  # all CMG joint rates
                ],
                dim=-1,
            ),
        }

    def _prepare_ctrl(self, action):
        rotor_ctrl = torch.full(
            (*self.batch_size, self.N_GIMBALS),
            self.ROTOR_SPEED,
            dtype=self.dtype,
            device=self.device,
        )
        return torch.cat([action, rotor_ctrl], dim=-1)

    def _make_batch(self, n):
        batch = super()._make_batch(n)
        rotor_vel_indices = [6 + 2 * i + 1 for i in range(self.N_GIMBALS)]
        batch.qvel[..., rotor_vel_indices] = self.ROTOR_SPEED
        return batch

    def _compute_reward(self, qpos_before, action):
        qx = self._dx.qpos[..., 4]
        qy = self._dx.qpos[..., 5]
        sun_alignment = 1.0 - 2.0 * (qx**2 + qy**2)

        ctrl_cost = self.CTRL_COST_WEIGHT * (action**2).sum(dim=-1)
        ang_vel = self._dx.qvel[..., 3:6]
        ang_vel_penalty = self.ANG_VEL_WEIGHT * (ang_vel**2).sum(dim=-1)

        reward = sun_alignment - ctrl_cost - ang_vel_penalty
        return reward.unsqueeze(-1).to(self.dtype)

    def _compute_terminated(self):
        return torch.zeros(
            *self.batch_size,
            1,
            dtype=torch.bool,
            device=self.device,
        )


@register_env("satellite_large")
class SatelliteLargeEnv(_SatelliteBase):
    """High-altitude satellite with 4 CMGs in a pyramid arrangement."""

    N_GIMBALS = 4
    ROTOR_SPEED = 100.0

    @classmethod
    def _xml_path(cls) -> str:
        return "satellite_large.xml"

    @classmethod
    def _camera_xml(cls) -> str:
        return '<camera name="side" pos="3 -3 2" xyaxes="0.707 0.707 0 -0.302 0.302 0.905" fovy="60"/>'


@register_env("satellite_small")
class SatelliteSmallEnv(_SatelliteBase):
    """Low-altitude CubeSat with 6 CMGs for redundant attitude control."""

    N_GIMBALS = 6
    ROTOR_SPEED = 200.0

    @classmethod
    def _xml_path(cls) -> str:
        return "satellite_small.xml"

    @classmethod
    def _camera_xml(cls) -> str:
        return '<camera name="side" pos="0.5 -0.5 0.3" xyaxes="0.707 0.707 0 -0.276 0.276 0.920" fovy="60"/>'
