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

    i = m.dof_tri_row_t
    j = m.dof_tri_col_t

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
    dof_Madr = m.dof_Madr_t
    qm[dof_Madr] = qm[dof_Madr] + d

    return qm


def full_m(m: Model, d: Data) -> torch.Tensor:
    """Reconstitute dense mass matrix from qM."""

    if not is_sparse(m):
        return d.qM

    i = m.dof_tri_row_t
    j = m.dof_tri_col_t

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
        device = tensor.device if isinstance(tensor, torch.Tensor) else None
        index = (
            torch.tensor([index], device=device).long()
            if scalar_index
            else torch.as_tensor(index).long().to(device=device)
        )

    is_batched = False
    if not torch.compiler.is_compiling() and is_batchedtensor(index):
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
    mask = mask[m.dof_bodyid_t] > 0

    index = vmap_compatible_index_select(m.body_rootid_t, dim=0, index=body_id).long()

    offset = point - vmap_compatible_index_select(d.subtree_com, dim=0, index=index)
    jacp = torch.vmap(lambda a, b=offset: a[3:] + math.cross(a[:3], b))(d.cdof)
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
        torch.arange(m.nbody, device=d.xipos.device),
    )
    return torch.sum(qfrc, axis=0)


def _muscle_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Quintic sigmoid: f(0)=f'(0)=f''(0)=0, f(1)=1, f'(1)=f''(1)=0."""
    sol = x * x * x * (3 * x * (2 * x - 5) + 10)
    sol = torch.where(x <= 0, 0.0, sol)
    sol = torch.where(x >= 1, 1.0, sol)
    return sol


def muscle_dynamics_timescale(
    dctrl: torch.Tensor,
    tau_act: torch.Tensor,
    tau_deact: torch.Tensor,
    smoothing_width: torch.Tensor,
) -> torch.Tensor:
    """Muscle time constant with optional smoothing."""
    tau_hard = torch.where(dctrl > 0, tau_act, tau_deact)
    tau_smooth = tau_deact + (tau_act - tau_deact) * _muscle_sigmoid(math.safe_div(dctrl, smoothing_width) + 0.5)
    return torch.where(smoothing_width < mujoco.mjMINVAL, tau_hard, tau_smooth)


def muscle_dynamics(ctrl: torch.Tensor, act: torch.Tensor, prm: torch.Tensor) -> torch.Tensor:
    """Muscle activation dynamics: da/dt."""
    ctrlclamp = torch.clamp(ctrl, 0, 1)
    actclamp = torch.clamp(act, 0, 1)

    tau_act = prm[0] * (0.5 + 1.5 * actclamp)
    tau_deact = prm[1] / (0.5 + 1.5 * actclamp)
    smoothing_width = prm[2]
    dctrl = ctrlclamp - act

    tau = muscle_dynamics_timescale(dctrl, tau_act, tau_deact, smoothing_width)
    return dctrl / torch.clamp_min(tau, mujoco.mjMINVAL)


def muscle_gain_length(length: torch.Tensor, lmin: torch.Tensor, lmax: torch.Tensor) -> torch.Tensor:
    """Normalized muscle length-gain curve."""
    a = 0.5 * (lmin + 1)
    b = 0.5 * (1 + lmax)

    out0 = 0.5 * torch.square((length - lmin) / torch.clamp_min(a - lmin, mujoco.mjMINVAL))
    out1 = 1 - 0.5 * torch.square((1 - length) / torch.clamp_min(1 - a, mujoco.mjMINVAL))
    out2 = 1 - 0.5 * torch.square((length - 1) / torch.clamp_min(b - 1, mujoco.mjMINVAL))
    out3 = 0.5 * torch.square((lmax - length) / torch.clamp_min(lmax - b, mujoco.mjMINVAL))

    out = torch.where(length <= b, out2, out3)
    out = torch.where(length <= 1, out1, out)
    out = torch.where(length <= a, out0, out)
    out = torch.where((lmin <= length) & (length <= lmax), out, 0.0)
    return out


def muscle_gain(
    length: torch.Tensor,
    vel: torch.Tensor,
    lengthrange: torch.Tensor,
    acc0: torch.Tensor,
    prm: torch.Tensor,
) -> torch.Tensor:
    """Muscle active force (gain)."""
    lrange = prm[:2]
    force, scale, lmin, lmax, vmax, _, fvmax = prm[2], prm[3], prm[4], prm[5], prm[6], prm[7], prm[8]

    force = torch.where(force < 0, scale / torch.clamp_min(acc0, mujoco.mjMINVAL), force)

    L0 = (lengthrange[1] - lengthrange[0]) / torch.clamp_min(lrange[1] - lrange[0], mujoco.mjMINVAL)
    L = lrange[0] + (length - lengthrange[0]) / torch.clamp_min(L0, mujoco.mjMINVAL)
    V = vel / torch.clamp_min(L0 * vmax, mujoco.mjMINVAL)

    FL = muscle_gain_length(L, lmin, lmax)

    y = fvmax - 1
    FV = torch.where(V <= y, fvmax - torch.square(y - V) / torch.clamp_min(y, mujoco.mjMINVAL), fvmax)
    FV = torch.where(V <= 0, torch.square(V + 1), FV)
    FV = torch.where(V <= -1, 0.0, FV)

    return -force * FL * FV


def muscle_bias(
    length: torch.Tensor,
    lengthrange: torch.Tensor,
    acc0: torch.Tensor,
    prm: torch.Tensor,
) -> torch.Tensor:
    """Muscle passive force (bias)."""
    lrange = prm[:2]
    force, scale, _, lmax, _, fpmax = prm[2], prm[3], prm[4], prm[5], prm[6], prm[7]

    force = torch.where(force < 0, scale / torch.clamp_min(acc0, mujoco.mjMINVAL), force)

    L0 = (lengthrange[1] - lengthrange[0]) / torch.clamp_min(lrange[1] - lrange[0], mujoco.mjMINVAL)
    L = lrange[0] + (length - lengthrange[0]) / torch.clamp_min(L0, mujoco.mjMINVAL)

    b = 0.5 * (1 + lmax)
    out1 = -force * fpmax * 0.5 * torch.square((L - 1) / torch.clamp_min(b - 1, mujoco.mjMINVAL))
    out2 = -force * fpmax * (0.5 + (L - b) / torch.clamp_min(b - 1, mujoco.mjMINVAL))

    out = torch.where(L <= b, out1, out2)
    out = torch.where(L <= 1, 0.0, out)
    return out
