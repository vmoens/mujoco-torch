# Copyright 2025 DeepMind Technologies Limited
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
"""Derivative functions."""

from typing import Optional

import torch
from mujoco_torch._src.types import BiasType
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import DynType
from mujoco_torch._src.types import GainType
from mujoco_torch._src.types import Model
from mujoco_torch._src.types import Option


def deriv_smooth_vel(m: Model, d: Data) -> Optional[torch.Tensor]:
  """Analytical derivative of smooth forces w.r.t. velocities."""
  qderiv = None

  # qDeriv += d qfrc_actuator / d qvel
  if not (m.opt.disableflags & DisableBit.ACTUATION):
    device = m.actuator_biasprm.device
    dtype = m.actuator_biasprm.dtype
    affine_bias = torch.tensor(
        m.actuator_biastype == BiasType.AFFINE,
        device=device,
        dtype=dtype,
    )
    bias_vel = m.actuator_biasprm[:, 2] * affine_bias
    affine_gain = torch.tensor(
        m.actuator_gaintype == GainType.AFFINE,
        device=device,
        dtype=dtype,
    )
    gain_vel = m.actuator_gainprm[:, 2] * affine_gain
    ctrl = d.ctrl.clone()
    dyn_mask = torch.tensor(
        m.actuator_dyntype != DynType.NONE,
        device=d.ctrl.device,
        dtype=torch.bool,
    )
    ctrl = torch.where(dyn_mask, d.act, ctrl)
    vel = bias_vel + gain_vel * ctrl
    actuator_moment = d.actuator_moment
    qderiv = actuator_moment.T @ (actuator_moment * vel.unsqueeze(1))

  # qDeriv += d qfrc_passive / d qvel
  if not (m.opt.disableflags & DisableBit.DAMPER):
    if qderiv is None:
      qderiv = -torch.diag(m.dof_damping)
    else:
      qderiv = qderiv - torch.diag(m.dof_damping)
  if m.ntendon:
    qderiv = qderiv - d.ten_J.T @ torch.diag(m.tendon_damping) @ d.ten_J

  if not (m.opt.disableflags & (DisableBit.DAMPER | DisableBit.SPRING)):
    if m.opt.has_fluid_params:
      raise NotImplementedError('fluid drag not supported for implicitfast')

  # TODO(robotics-simulation): rne derivative

  return qderiv
