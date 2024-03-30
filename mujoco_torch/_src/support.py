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
from typing import Tuple

import torch
# from torch import numpy as torch
from mujoco_torch._src import scan
# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import Model


# pylint: enable=g-importing-member

def vmap_compatible_index_select(tensor, dim, index):
  from torch._C._functorch import is_batchedtensor, _remove_batch_dim, _add_batch_dim, maybe_get_level

  is_batched = False
  if isinstance(index, torch.Tensor) and is_batchedtensor(index):
    # _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
    lvl = maybe_get_level(index)
    index = _remove_batch_dim(index, lvl, 0, 0)
    is_batched = True
  out = torch.index_select(tensor, dim, index)
  if is_batched:
    out = _add_batch_dim(out, 0, lvl)
  return out

def jac(
    m: Model, d: Data, point: torch.Tensor, body_id: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Compute pair of (NV, 3) Jacobians of global point attached to body."""
  fn = lambda carry, b: b if carry is None else b + carry
  mask = (torch.arange(m.nbody) == body_id) * 1
  mask = scan.body_tree(m, fn, 'b', 'b', mask, reverse=True)
  mask = mask[torch.tensor(m.dof_bodyid)] > 0

  index = vmap_compatible_index_select(torch.tensor(m.body_rootid), dim=0, index=body_id).long()

  offset = point - vmap_compatible_index_select(d.subtree_com, dim=0, index=index)
  jacp = torch.vmap(lambda a, b=offset: a[3:] + torch.linalg.cross(a[:3], b))(d.cdof)
  jacp = torch.vmap(torch.multiply)(jacp, mask)
  jacr = torch.vmap(torch.multiply)(d.cdof[:, :3], mask)

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
  return jacp @ force + jacr @ torque


def xfrc_accumulate(m: Model, d: Data) -> torch.Tensor:
  """Accumulate xfrc_applied into a qfrc."""
  qfrc = torch.vmap(apply_ft, (None, None, 0, 0, 0, 0))(
      m,
      d,
      d.xfrc_applied[:, :3],
      d.xfrc_applied[:, 3:],
      d.xipos,
      torch.arange(m.nbody),
  )
  return torch.sum(qfrc, axis=0)
