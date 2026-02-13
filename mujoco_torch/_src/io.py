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
"""Functions to initialize, load, or save data."""

from typing import Optional, Union

import mujoco
import numpy as np
import torch
from mujoco_torch._src import constraint
from mujoco_torch._src import device
from mujoco_torch._src.types import Contact
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import Model

DEFAULT_DTYPE = torch.float64


def _make_data_public_fields(m: Model) -> dict:
  """Create public fields for the Data object."""
  zero_fields = {
      'time': (),
      'qvel': (m.nv,),
      'act': (m.na,),
      'qacc_warmstart': (m.nv,),
      'ctrl': (m.nu,),
      'qfrc_applied': (m.nv,),
      'xfrc_applied': (m.nbody, 6),
      'mocap_pos': (m.nmocap, 3),
      'mocap_quat': (m.nmocap, 4),
      'qacc': (m.nv,),
      'act_dot': (m.na,),
      'userdata': (getattr(m, 'nuserdata', 0),),
      'sensordata': (m.nsensordata,),
      'xpos': (m.nbody, 3),
      'xquat': (m.nbody, 4),
      'xmat': (m.nbody, 3, 3),
      'xipos': (m.nbody, 3),
      'ximat': (m.nbody, 3, 3),
      'xanchor': (m.njnt, 3),
      'xaxis': (m.njnt, 3),
      'geom_xpos': (m.ngeom, 3),
      'geom_xmat': (m.ngeom, 3, 3),
      'site_xpos': (m.nsite, 3),
      'site_xmat': (m.nsite, 3, 3),
      'cam_xpos': (m.ncam, 3),
      'cam_xmat': (m.ncam, 3, 3),
      'subtree_com': (m.nbody, 3),
      'actuator_force': (m.nu,),
      'actuator_length': (m.nu,),
      'qfrc_bias': (m.nv,),
      'qfrc_gravcomp': (m.nv,),
      'qfrc_fluid': (m.nv,),
      'qfrc_passive': (m.nv,),
      'qfrc_actuator': (m.nv,),
      'qfrc_smooth': (m.nv,),
      'qacc_smooth': (m.nv,),
      'qfrc_constraint': (m.nv,),
      'qfrc_inverse': (m.nv,),
      'cvel': (m.nbody, 6),
      'cdof': (m.nv, 6),
      'cdof_dot': (m.nv, 6),
      'ten_length': (m.ntendon,),
  }
  return {
      k: torch.zeros(v, dtype=DEFAULT_DTYPE)
      for k, v in zero_fields.items()
  }


def _make_data_contact(ncon: int, contact_dim: torch.Tensor, efc_address: torch.Tensor) -> Contact:
  """Create contact for the Data object."""
  return Contact(
      dist=torch.zeros((ncon,), dtype=DEFAULT_DTYPE),
      pos=torch.zeros((ncon, 3), dtype=DEFAULT_DTYPE),
      frame=torch.zeros((ncon, 3, 3), dtype=DEFAULT_DTYPE),
      includemargin=torch.zeros((ncon,), dtype=DEFAULT_DTYPE),
      friction=torch.zeros((ncon, 5), dtype=DEFAULT_DTYPE),
      solref=torch.zeros((ncon, mujoco.mjNREF), dtype=DEFAULT_DTYPE),
      solreffriction=torch.zeros((ncon, mujoco.mjNREF), dtype=DEFAULT_DTYPE),
      solimp=torch.zeros((ncon, mujoco.mjNIMP), dtype=DEFAULT_DTYPE),
      contact_dim=contact_dim,
      geom1=torch.full((ncon,), -1, dtype=torch.int64),
      geom2=torch.full((ncon,), -1, dtype=torch.int64),
      geom=torch.full((ncon, 2), -1, dtype=torch.int64),
      efc_address=efc_address,
      batch_size=[ncon],
  )


def make_data(m: Union[Model, mujoco.MjModel]) -> Data:
  """Allocate and initialize Data.

  Args:
    m: the model to use (mujoco_torch Model or mujoco MjModel)

  Returns:
    an initialized Data with all fields set to zeros of appropriate shapes.
  """
  # For MjModel, convert to our Model first
  if isinstance(m, mujoco.MjModel):
    m = put_model(m)

  # Get constraint counts purely from Model (no Data needed).
  ne, nf, nl, ncon, nefc = constraint.constraint_sizes(m)
  ns = ne + nf + nl

  # Build contact dim and efc_address (mujoco_torch uses contact_dim=3 for all, 4 efc rows per contact)
  contact_dim = torch.full((ncon,), 3, dtype=torch.int32) if ncon > 0 else torch.zeros((0,), dtype=torch.int32)
  efc_address = (
      torch.arange(ns, ns + ncon * 4, 4, dtype=torch.int32)
      if ncon > 0
      else torch.zeros((0,), dtype=torch.int32)
  )
  contact = _make_data_contact(ncon, contact_dim, efc_address)

  # Build public fields
  public_fields = _make_data_public_fields(m)
  public_fields['qpos'] = torch.as_tensor(m.qpos0, dtype=DEFAULT_DTYPE)
  public_fields['eq_active'] = torch.as_tensor(m.eq_active0, dtype=torch.int32)

  # Build zero fields for impl-specific data
  zero_impl = {
      'solver_niter': torch.tensor(0, dtype=torch.int32),
      'cinert': torch.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      'ten_wrapadr': torch.zeros(m.ntendon, dtype=torch.int32),
      'ten_wrapnum': torch.zeros(m.ntendon, dtype=torch.int32),
      'ten_J': torch.zeros((m.ntendon, m.nv), dtype=DEFAULT_DTYPE),
      'wrap_obj': torch.zeros((m.nwrap, 2), dtype=torch.int32),
      'wrap_xpos': torch.zeros((m.nwrap, 6), dtype=DEFAULT_DTYPE),
      'actuator_moment': torch.zeros((m.nu, m.nv), dtype=DEFAULT_DTYPE),
      'crb': torch.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      'qM': torch.zeros(m.nM, dtype=DEFAULT_DTYPE),
      'qLD': torch.zeros(m.nM, dtype=DEFAULT_DTYPE),
      'qLDiagInv': torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      'ten_velocity': torch.zeros(m.ntendon, dtype=DEFAULT_DTYPE),
      'actuator_velocity': torch.zeros(m.nu, dtype=DEFAULT_DTYPE),
      'cacc': torch.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      'cfrc_int': torch.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      'cfrc_ext': torch.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      'subtree_linvel': torch.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      'subtree_angmom': torch.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      'efc_J': torch.zeros((nefc, m.nv), dtype=DEFAULT_DTYPE),
      'efc_pos': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_margin': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_frictionloss': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_D': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_aref': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_force': torch.zeros(nefc, dtype=DEFAULT_DTYPE),
      'efc_type': torch.zeros(nefc, dtype=torch.int32),
  }

  d = Data(
      ne=torch.tensor(ne, dtype=torch.int32),
      nf=torch.tensor(nf, dtype=torch.int32),
      nl=torch.tensor(nl, dtype=torch.int32),
      nefc=torch.tensor(nefc, dtype=torch.int32),
      ncon=torch.tensor(ncon, dtype=torch.int32),
      contact=contact,
      **public_fields,
      **zero_impl,
      batch_size=[],
  )

  # Set mocap_pos/quat = body_pos/quat for mocap bodies (as done in C MuJoCo)
  if m.nmocap > 0:
    body_mask = m.body_mocapid >= 0
    if np.any(body_mask):
      body_pos = torch.as_tensor(m.body_pos, dtype=DEFAULT_DTYPE)
      body_quat = torch.as_tensor(m.body_quat, dtype=DEFAULT_DTYPE)
      mocapid = m.body_mocapid[body_mask]
      mocap_pos = torch.zeros((m.nmocap, 3), dtype=DEFAULT_DTYPE)
      mocap_quat = torch.zeros((m.nmocap, 4), dtype=DEFAULT_DTYPE)
      mocap_pos[mocapid] = body_pos[body_mask]
      mocap_quat[mocapid] = body_quat[body_mask]
      d = d.replace(mocap_pos=mocap_pos, mocap_quat=mocap_quat)

  return d


def put_model(m: mujoco.MjModel, device_target: Optional[torch.device] = None) -> Model:
  """Converts mujoco.MjModel to mujoco_torch Model.

  Args:
    m: the mujoco MjModel to convert
    device_target: optional torch device (currently ignored; tensors stay on default device)

  Returns:
    a mujoco_torch Model with tensors for all array fields.
  """
  del device_target  # device placement can be done by caller if needed
  return device.device_put(m)
