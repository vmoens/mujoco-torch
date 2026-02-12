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
from mujoco_torch._src.device import device_get_into
from mujoco_torch._src.device import device_put

# Forward dynamics
from mujoco_torch._src.forward import euler
from mujoco_torch._src.forward import forward
from mujoco_torch._src.forward import fwd_acceleration
from mujoco_torch._src.forward import fwd_actuation
from mujoco_torch._src.forward import fwd_position
from mujoco_torch._src.forward import fwd_velocity
from mujoco_torch._src.forward import rungekutta4
from mujoco_torch._src.forward import step

# Inverse dynamics
from mujoco_torch._src.inverse import inverse

# I/O
from mujoco_torch._src.io import make_data

# Passive forces
from mujoco_torch._src.passive import passive

# Raycasting
from mujoco_torch._src.ray import ray
from mujoco_torch._src.ray import ray_geom

# Sensors
from mujoco_torch._src.sensor import sensor_acc
from mujoco_torch._src.sensor import sensor_pos
from mujoco_torch._src.sensor import sensor_vel

# Smooth dynamics
from mujoco_torch._src.smooth import com_pos
from mujoco_torch._src.smooth import com_vel
from mujoco_torch._src.smooth import crb
from mujoco_torch._src.smooth import factor_m
from mujoco_torch._src.smooth import kinematics
from mujoco_torch._src.smooth import mul_m
from mujoco_torch._src.smooth import rne
from mujoco_torch._src.smooth import solve_m
from mujoco_torch._src.smooth import transmission

# Solver
from mujoco_torch._src.solver import solve

# Support
from mujoco_torch._src.support import apply_ft
from mujoco_torch._src.support import full_m
from mujoco_torch._src.support import jac
from mujoco_torch._src.support import xfrc_accumulate

# Types (public)
from mujoco_torch._src.types import BiasType
from mujoco_torch._src.types import CamLightType
from mujoco_torch._src.types import ConeType
from mujoco_torch._src.types import ConstraintType
from mujoco_torch._src.types import Contact
from mujoco_torch._src.types import ConvexMesh
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import DynType
from mujoco_torch._src.types import EnableBit
from mujoco_torch._src.types import EqType
from mujoco_torch._src.types import GainType
from mujoco_torch._src.types import GeomType
from mujoco_torch._src.types import IntegratorType
from mujoco_torch._src.types import JacobianType
from mujoco_torch._src.types import JointType
from mujoco_torch._src.types import Model
from mujoco_torch._src.types import ObjType
from mujoco_torch._src.types import Option
from mujoco_torch._src.types import SensorType
from mujoco_torch._src.types import SolverType
from mujoco_torch._src.types import Statistic
from mujoco_torch._src.types import TrnType
from mujoco_torch._src.types import WrapType
