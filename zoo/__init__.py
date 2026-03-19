"""mujoco-torch environment zoo.

A collection of standard MuJoCo locomotion environments implemented as batched
TorchRL ``EnvBase`` subclasses backed by mujoco-torch.

Usage::

    from zoo import ENVS
    env = ENVS["halfcheetah"](num_envs=64)
"""

from zoo.ant import AntEnv
from zoo.base import MujocoTorchEnv
from zoo.cartpole import CartPoleEnv
from zoo.halfcheetah import HalfCheetahEnv
from zoo.hopper import HopperEnv
from zoo.satellite import SatelliteLargeEnv, SatelliteSmallEnv
from zoo.swimmer import SwimmerEnv
from zoo.walker2d import Walker2dEnv

ENVS: dict[str, type[MujocoTorchEnv]] = {
    "halfcheetah": HalfCheetahEnv,
    "ant": AntEnv,
    "hopper": HopperEnv,
    "walker2d": Walker2dEnv,
    "swimmer": SwimmerEnv,
    "cartpole": CartPoleEnv,
    "satellite_large": SatelliteLargeEnv,
    "satellite_small": SatelliteSmallEnv,
}

__all__ = [
    "ENVS",
    "MujocoTorchEnv",
    "HalfCheetahEnv",
    "AntEnv",
    "HopperEnv",
    "Walker2dEnv",
    "SwimmerEnv",
    "CartPoleEnv",
    "SatelliteLargeEnv",
    "SatelliteSmallEnv",
]
