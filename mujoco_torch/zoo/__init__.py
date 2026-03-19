"""mujoco-torch environment zoo.

A collection of standard MuJoCo environments implemented as batched
TorchRL ``EnvBase`` subclasses backed by mujoco-torch.

Usage::

    from mujoco_torch.zoo import ENVS
    env = ENVS["halfcheetah"](num_envs=64)

New environments are discovered automatically: decorate the class with
``@register_env("name")`` and it will appear in ``ENVS`` on import.
"""

import importlib
import pkgutil

from mujoco_torch.zoo.base import ENVS, MujocoTorchEnv, register_env

# Auto-import all sibling modules to trigger @register_env decorators.
for _info in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(_info.name)

# Re-export every registered env class so `from mujoco_torch.zoo import XxxEnv` works.
globals().update({cls.__name__: cls for cls in ENVS.values()})

__all__ = ["ENVS", "MujocoTorchEnv", "register_env", *[cls.__name__ for cls in ENVS.values()]]
