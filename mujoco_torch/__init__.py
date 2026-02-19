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

# Passive forces
from mujoco_torch._src.passive import passive

# Raycasting
from mujoco_torch._src.ray import ray, ray_geom

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
