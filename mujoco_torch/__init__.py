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

_apply_patches()
del _apply_patches

# pylint:disable=g-importing-member

# Collision
from mujoco_torch._src.collision_driver import collision

# Constraint
from mujoco_torch._src.constraint import make_constraint

# Derivative
from mujoco_torch._src.derivative import deriv_smooth_vel

# Device (put/get model and data)
from mujoco_torch._src.device import device_get_into, device_put

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

# tensordict picks its UnbatchedTensor implementation at import time by
# inspecting MetaConverter's *on-disk* source.  Our in-memory patch isn't
# visible to inspect.getsource, so tensordict may have chosen the wrong
# implementation.  Fix it up now that both patches and tensordict are loaded.
from mujoco_torch.patches import fix_tensordict_unbatched as _fix_ut

_fix_ut()
del _fix_ut
