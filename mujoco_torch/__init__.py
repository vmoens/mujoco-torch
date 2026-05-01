# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Public API for mujoco-torch (MJX ported to PyTorch)."""

# Apply monkey-patches for upstream PyTorch PRs that haven't landed yet.
# Safe to call unconditionally: each patch is a no-op when the fix is present.
# Patches MUST run before tensordict is imported (the _src imports below
# trigger tensordict loading) so that tensordict's runtime guard sees the
# patched MetaConverter and selects the right UnbatchedTensor implementation.
from mujoco_torch.patches import apply as _apply_patches
from mujoco_torch.patches import fix_tensordict_unbatched as _fix_ut

_apply_patches()
# Force tensordict to pick the wrapper-subclass UnbatchedTensor *before* any
# _src module below binds ``from tensordict import UnbatchedTensor``.  If we
# ran this after the _src imports (as the code used to), each _src module
# would cache the pre-patch class while tensordict internals use the
# post-patch class, yielding two coexisting UnbatchedTensor classes with
# identical names but different type ids — which trips Dynamo's type_id
# guard on call 2 of compile(vmap(step)) and forces a recompile.
_fix_ut()
del _apply_patches
del _fix_ut

# pylint:disable=g-importing-member

# Differentiable mode (experimental)
# Collision
from mujoco_torch._src.collision_driver import collision

# Constraint
from mujoco_torch._src.constraint import make_constraint

# Derivative
from mujoco_torch._src.derivative import deriv_smooth_vel

# Device (put/get model and data)
from mujoco_torch._src.device import device_get_into, device_put
from mujoco_torch._src.diff_config import DiffConfig, differentiable_mode, get_diff_config

# Forward dynamics
from mujoco_torch._src.forward import (
    euler,
    forward,
    fwd_acceleration,
    fwd_actuation,
    fwd_position,
    fwd_velocity,
    rungekutta4,
    step,
)

# Inverse dynamics
from mujoco_torch._src.inverse import inverse

# I/O
from mujoco_torch._src.io import make_data

# Logging
from mujoco_torch._src.log import logger as mujoco_logger

# Math (selected -- the rest of `mujoco_torch._src.math` is internal)
from mujoco_torch._src.math import random_unit_quat

# Passive forces
from mujoco_torch._src.passive import passive

# Raycasting
from mujoco_torch._src.ray import ray, ray_geom

# Rendering
from mujoco_torch._src.render import precompute_render_data, render, render_batch

# Sensors
from mujoco_torch._src.sensor import sensor_acc, sensor_pos, sensor_vel

# Smooth dynamics
from mujoco_torch._src.smooth import (
    com_pos,
    com_vel,
    crb,
    factor_m,
    kinematics,
    mul_m,
    rne,
    solve_m,
    tendon,
    tendon_armature,
    transmission,
)

# Solver
from mujoco_torch._src.solver import solve

# Support
from mujoco_torch._src.support import apply_ft, full_m, jac, xfrc_accumulate

# Types (public)
from mujoco_torch._src.types import (
    BiasType,
    CamLightType,
    ConeType,
    ConstraintType,
    Contact,
    ConvexMesh,
    Data,
    DisableBit,
    DynType,
    EnableBit,
    EqType,
    GainType,
    GeomType,
    IntegratorType,
    JacobianType,
    JointType,
    Model,
    ObjType,
    Option,
    SensorType,
    SolverType,
    Statistic,
    TrnType,
    WrapType,
)
