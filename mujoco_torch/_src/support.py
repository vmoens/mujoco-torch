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

import mujoco
import numpy as np
import torch
from torch._C._functorch import _add_batch_dim, _remove_batch_dim, is_batchedtensor, maybe_get_level

from mujoco_torch._src import math, scan

# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data, JacobianType, Model

# pylint: enable=g-importing-member


def to_int(x) -> int:
    """Extract a Python int from a tensor that may be inside torch.vmap.

    Inside vmap, calling ``int(tensor)`` / ``.item()`` is not supported.
    This helper flattens the tensor first and takes element [0], which is
    safe because the values wrapped here (ne, nf, ncon, …) are model-level
    constants identical across all batch elements.
    """
    if isinstance(x, int):
        return x
    return int(x.flatten()[0])


def is_sparse(m: mujoco.MjModel | Model) -> bool:
    """Return True if this model should create sparse mass matrices."""
    if m.opt.jacobian == JacobianType.AUTO:
        return m.nv >= 60
    return m.opt.jacobian == JacobianType.SPARSE


def make_m(
    m: Model,
    a: torch.Tensor,
    b: torch.Tensor,
    d: torch.Tensor | None = None,
) -> torch.Tensor:
    """Computes M = a @ b.T + diag(d)."""

    i = torch.as_tensor(m.dof_tri_row, device=a.device)
    j = torch.as_tensor(m.dof_tri_col, device=a.device)

    if not is_sparse(m):
        qm = a @ b.T
        if d is not None:
            qm = qm + torch.diag(d)
        mask = torch.zeros((m.nv, m.nv), dtype=torch.bool, device=a.device)
        mask[(i, j)] = True
        qm = qm * mask
        qm = qm + torch.tril(qm, -1).T
        return qm

    a_i = a[i]
    b_j = b[j]
    qm = torch.vmap(torch.dot)(a_i, b_j)

    if d is not None:
        qm = qm.clone()
        dof_Madr = torch.as_tensor(m.dof_Madr)
        qm[dof_Madr] = qm[dof_Madr] + d

    return qm


def full_m(m: Model, d: Data) -> torch.Tensor:
    """Reconstitute dense mass matrix from qM."""

    if not is_sparse(m):
        return d.qM

    i = torch.as_tensor(m.dof_tri_row, device=d.qM.device)
    j = torch.as_tensor(m.dof_tri_col, device=d.qM.device)

    mat = torch.zeros((m.nv, m.nv), dtype=d.qM.dtype, device=d.qM.device)
    mat[(i, j)] = d.qM
    mat = mat + torch.triu(mat, 1).T

    return mat


def local_to_global(
    world_pos: torch.Tensor,
    world_quat: torch.Tensor,
    local_pos: torch.Tensor,
    local_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts local position/orientation to world frame."""
    pos = world_pos + math.rotate(local_pos, world_quat)
    mat = math.quat_to_mat(math.quat_mul(world_quat, local_quat))
    return pos, mat


def vmap_compatible_index_select(tensor, dim, index):
    scalar_index = not isinstance(index, torch.Tensor) and isinstance(index, (int, np.integer))
    if not isinstance(index, torch.Tensor):
        index = torch.tensor([index]).long() if scalar_index else torch.as_tensor(index).long()

    is_batched = False
    if is_batchedtensor(index):
        # _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)
        lvl = maybe_get_level(index)
        index = _remove_batch_dim(index, lvl, 0, 0)
        is_batched = True

    squeeze_out = not is_batched and index.ndim == 0
    if index.ndim == 0:
        index = index.unsqueeze(0)
    out = torch.index_select(tensor, dim, index)
    if is_batched:
        out = _add_batch_dim(out, 0, lvl)
    elif scalar_index or squeeze_out:
        out = out.squeeze(dim)
    return out


def jac(m: Model, d: Data, point: torch.Tensor, body_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute pair of (NV, 3) Jacobians of global point attached to body."""
    fn = lambda carry, b: b if carry is None else b + carry
    device = point.device if isinstance(point, torch.Tensor) else None
    mask = (torch.arange(m.nbody, device=device) == body_id) * 1
    mask = scan.body_tree(m, fn, "b", "b", mask, reverse=True)
    mask = mask[torch.as_tensor(m.dof_bodyid)] > 0

    index = vmap_compatible_index_select(torch.as_tensor(m.body_rootid), dim=0, index=body_id).long()

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
    force = force.to(jacp.dtype)
    torque = torque.to(jacr.dtype)
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
