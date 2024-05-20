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
"""Engine support functions."""
from typing import Optional, Tuple, Union

import mujoco
from mujoco_torch._src import math
from mujoco_torch._src import scan
from mujoco_torch._src.smooth import _set_at
# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import JacobianType
from mujoco_torch._src.types import Model
# pylint: enable=g-importing-member
import torch

from torch._C._functorch import is_batchedtensor, _remove_batch_dim,dlevel,maybe_get_level, current_level, _add_batch_dim, get_unwrapped, _vmap_increment_nesting

def is_sparse(m: Union[mujoco.MjModel, Model]) -> bool:
  """Return True if this model should create sparse mass matrices.

  Args:
    m: a MuJoCo or MJX model

  Returns:
    True if provided model should create sparse mass matrices

  Modern TPUs have specialized hardware for rapidly operating over sparse
  matrices, whereas GPUs tend to be faster with dense matrices as long as they
  fit onto the device.  As such, the default behavior in MJX (via
  ``JacobianType.AUTO``) is sparse if ``nv`` is >= 60 or MJX detects a TPU as
  the default backend, otherwise dense.
  """
  # AUTO is a rough heuristic - you may see better performance for your workload
  # and compute by explicitly setting jacobian to dense or sparse
  # return False
  if m.opt.jacobian == JacobianType.AUTO:
    return m.nv >= 60 or torch.get_default_device().type == 'cuda'
  return m.opt.jacobian == JacobianType.SPARSE


def make_m(
    m: Model, a: torch.Tensor, b: torch.Tensor, d: Optional[torch.Tensor] = None
) -> torch.Tensor:
  """Computes M = a @ b.T + diag(d)."""

  ij = []
  for i in range(m.nv):
    j = i
    while j > -1:
      ij.append((i, j))
      j = m.dof_parentid[j]

  i, j = (torch.tensor(x) for x in zip(*ij))

  if not is_sparse(m):
    qm = a @ b.T
    if d is not None:
      qm += torch.diag(d)
    mask = _set_at(torch.zeros((m.nv, m.nv), dtype=torch.bool), (i, j), True)
    qm = qm * mask
    qm = qm + torch.tril(qm, -1).T
    return qm

  a_i = a[i]
  b_j = b[j]
  qm = torch.vmap(torch.dot)(a_i, b_j)

  # add diagonal
  if d is not None:
    qm = qm.at[m.dof_Madr].add(d)

  return qm


def full_m(m: Model, d: Data) -> torch.Tensor:
  """Reconstitute dense mass matrix from qM."""

  if not is_sparse(m):
    return d.qM

  ij = []
  for i in range(m.nv):
    j = i
    while j > -1:
      ij.append((i, j))
      j = m.dof_parentid[j]

  i, j = (torch.tensor(x) for x in zip(*ij))

  mat = torch.zeros((m.nv, m.nv)).at[(i, j)].set(d.qM)

  # also set upper triangular
  mat = mat + torch.tril(mat, -1).T

  return mat


def mul_m(m: Model, d: Data, vec: torch.Tensor) -> torch.Tensor:
  """Multiply vector by inertia matrix."""

  if not is_sparse(m):
    return d.qM @ vec

  diag_mul = d.qM[torch.tensor(m.dof_Madr)] * vec

  is_, js, madr_ijs = [], [], []
  for i in range(m.nv):
    madr_ij, j = m.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  i, j, madr_ij = (torch.tensor(x, dtype=torch.int32) for x in (is_, js, madr_ijs))

  out = diag_mul.at[i].add(d.qM[madr_ij] * vec[j])
  out = out.at[j].add(d.qM[madr_ij] * vec[i])

  return out

def vmap_compatible_index_select(tensor, dim, index):
  # levels = []
  # while isinstance(index, torch.Tensor) and is_batchedtensor(index):
  #   levels.append(maybe_get_level(index))
  #   index = get_unwrapped(index)
  # out = torch.index_select(tensor, dim, index)
  # for level in levels[::-1]:
  #   out = _add_batch_dim(out, 0, level)
  # return out
  unsq = index.ndim == 0
  if unsq:
    unsq = True
    index = index.unsqueeze(0)
  result = torch.index_select(tensor, dim, index)
  if unsq:
    result = result.squeeze(dim)
  return result


def expand_right(tensor, shape):
  while tensor.ndim < len(shape):
    tensor = tensor.unsqueeze(-1)
  return tensor.expand(shape)
def jac(
    m: Model, d: Data, point: torch.Tensor, body_id: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute pair of (NV, 3) Jacobians of global point attached to body."""
  fn = lambda carry, b: b if carry is None else b + carry
  mask = (torch.arange(m.nbody) == body_id) * 1
  mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  mask = mask[torch.tensor(m.dof_bodyid)] > 0

  # vmapping over index_select is broken
  index = vmap_compatible_index_select(m.body_rootid, dim=0, index=body_id.long()).long()
  offset = point - vmap_compatible_index_select(d.subtree_com, dim=0, index=index)
  def inner(a, b=offset):
    return a[3:] + torch.linalg.cross(a[:3], b)
  jacp = torch.vmap(inner)(d.cdof)
  jacp = torch.where(expand_right(mask, jacp.shape), jacp, 0)
  dcdof = d.cdof[:, :3]
  jacr = torch.where(expand_right(mask, dcdof.shape), dcdof, 0)

  return jacp, jacr


def jac_dif_pair(
    m: Model,
    d: Data,
    pos: torch.Tensor,
    body_1: torch.Tensor,
    body_2: torch.Tensor,
) -> torch.Tensor:
  """Compute Jacobian difference for two body points."""
  jacp2, _ = jac(m, d, pos, body_2)
  jacp1, _ = jac(m, d, pos, body_1)
  return jacp2 - jacp1


def apply_ft(
    m: Model,
    d: Data,
    force: torch.Tensor,
    torque: torch.Tensor,
    point: torch.Tensor,
    body_id: torch.Tensor,
) -> torch.Tensor:
  """Apply Cartesian force and torque."""
  jacp, jacr = jac(m, d, point, body_id)
  # [11, 7, 3] x [3] + [11, 3] x [3]
  def inner(a, b):
    return a.dot(force) + b.dot(torque)
  result = torch.vmap(inner, (0, 0), (0, ))(jacp, jacr)
  return result


def xfrc_accumulate(m: Model, d: Data) -> torch.Tensor:
  """Accumulate xfrc_applied into a qfrc."""
  # d0 = dims(1)
  # qfrc = apply_ft(
  #     m,
  #     d,
  #     d.xfrc_applied[d0, :3],
  #     d.xfrc_applied[d0, 3:],
  #     d.xipos[d0],
  #     torch.arange(m.nbody)[d0],
  # )
  # return qfrc.sum(d0)
  qfrc = torch.vmap(apply_ft, (None, None, 0, 0, 0, 0))(
      m,
      d,
      d.xfrc_applied[:, :3],
      d.xfrc_applied[:, 3:],
      d.xipos,
      torch.arange(m.nbody),
  )
  return qfrc.sum(0)


def local_to_global(
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    local_pos: torch.Tensor,
    local_quat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Converts local position/orientation to world frame."""
  pos = world_pos + math.rotate(local_pos, world_quat)
  mat = math.quat_to_mat(math.quat_mul(world_quat, local_quat))
  return pos, mat


def get_custom_numeric(m: Union[Model, mujoco.MjModel], name: str) -> float:
  """Returns a custom numeric given an MjModel or mujoco_torch.Model."""
  for i in range(m.nnumeric):
    name_ = m.obj_names[m.name_numericadr[i] :].decode('utf-8').split('\x00', 1)[0]
    if name_ == name:
      return m.numeric_data[m.numeric_adr[i]]

  return -1
