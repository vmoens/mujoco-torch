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

# pylint: enable=g-importing-member
import numpy as np
# from jax import numpy as torch
import torch
from mujoco_torch._src import collision_driver
from mujoco_torch._src import constraint
# pylint: disable=g-importing-member
from mujoco_torch._src.types import Contact
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import Model

DEFAULT_DTYPE = None

# @torch.compiler.disable()
def make_data(m: Model) -> Data:
  """Allocate and initialize Data."""

  # create first d to get num contacts and nc
  d = Data(
      # solver_niter=torch.tensor(0, dtype=torch.int32),
      solver_niter=torch.tensor(0, dtype=torch.int32),
      ne=0,
      nf=0,
      nl=0,
      nefc=0,
      ncon=0,
      time=torch.zeros((), dtype=DEFAULT_DTYPE),
      qpos=m.qpos0,
      qvel=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      act=torch.zeros(m.na, dtype=DEFAULT_DTYPE),
      qacc_warmstart=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      ctrl=torch.zeros(m.nu, dtype=DEFAULT_DTYPE),
      qfrc_applied=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      xfrc_applied=torch.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      eq_active=torch.zeros(m.neq, dtype=torch.int32),
      qacc=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      act_dot=torch.zeros(m.na, dtype=DEFAULT_DTYPE),
      xpos=torch.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      xquat=torch.zeros((m.nbody, 4), dtype=DEFAULT_DTYPE),
      xmat=torch.zeros((m.nbody, 3, 3), dtype=DEFAULT_DTYPE),
      xipos=torch.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      ximat=torch.zeros((m.nbody, 3, 3), dtype=DEFAULT_DTYPE),
      xanchor=torch.zeros((m.njnt, 3), dtype=DEFAULT_DTYPE),
      xaxis=torch.zeros((m.njnt, 3), dtype=DEFAULT_DTYPE),
      geom_xpos=torch.zeros((m.ngeom, 3), dtype=DEFAULT_DTYPE),
      geom_xmat=torch.zeros((m.ngeom, 3, 3), dtype=DEFAULT_DTYPE),
      subtree_com=torch.zeros((m.nbody, 3), dtype=DEFAULT_DTYPE),
      cdof=torch.zeros((m.nv, 6), dtype=DEFAULT_DTYPE),
      cinert=torch.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      actuator_length=torch.zeros(m.nu, dtype=DEFAULT_DTYPE),
      actuator_moment=torch.zeros((m.nu, m.nv), dtype=DEFAULT_DTYPE),
      crb=torch.zeros((m.nbody, 10), dtype=DEFAULT_DTYPE),
      qM=torch.zeros(m.nM, dtype=DEFAULT_DTYPE),
      qLD=torch.zeros(m.nM, dtype=DEFAULT_DTYPE),
      qLDiagInv=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qLDiagSqrtInv=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      contact=Contact.zero(),
      efc_J=torch.zeros((), dtype=DEFAULT_DTYPE),
      efc_frictionloss=torch.zeros((), dtype=DEFAULT_DTYPE),
      efc_D=torch.zeros((), dtype=DEFAULT_DTYPE),
      actuator_velocity=torch.zeros(m.nu, dtype=DEFAULT_DTYPE),
      cvel=torch.zeros((m.nbody, 6), dtype=DEFAULT_DTYPE),
      cdof_dot=torch.zeros((m.nv, 6), dtype=DEFAULT_DTYPE),
      qfrc_bias=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_passive=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      efc_aref=torch.zeros((), dtype=DEFAULT_DTYPE),
      actuator_force=torch.zeros(m.nu, dtype=DEFAULT_DTYPE),
      qfrc_actuator=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_smooth=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qacc_smooth=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_constraint=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      qfrc_inverse=torch.zeros(m.nv, dtype=DEFAULT_DTYPE),
      efc_force=torch.zeros((), dtype=DEFAULT_DTYPE),
  )

  # get contact data with correct shapes
  ncon = collision_driver.ncon(m)
  d = d.replace(contact=Contact.zero((ncon,)), ncon=ncon)
  d = d.tree_replace({'contact.dim': 3 * np.ones(ncon)})

  ne, nf, nl, nc = constraint.count_constraints(m, d)
  d = d.replace(ne=ne, nf=nf, nl=nl, nefc=ne + nf + nl + nc)
  ns = ne + nf + nl
  d = d.tree_replace({'contact.efc_address': np.arange(ns, ns + ncon * 4, 4)})
  d = d.replace(
      efc_J=torch.zeros((d.nefc, m.nv), dtype=DEFAULT_DTYPE),
      efc_frictionloss=torch.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_D=torch.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_aref=torch.zeros(d.nefc, dtype=DEFAULT_DTYPE),
      efc_force=torch.zeros(d.nefc, dtype=DEFAULT_DTYPE),
  )

  return d
