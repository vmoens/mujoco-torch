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
"""Core smooth dynamics functions."""

import mujoco
import numpy as np
import torch

from mujoco_torch._src import math, scan, support
from mujoco_torch._src.math import _CachedConst

# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data, DisableBit, JointType, Model, TrnType, WrapType

# pylint: enable=g-importing-member

# Inertia triu extraction: diag first, then off-diag (matches inert_mul's tri_id)
_INERT_TRIU_I = _CachedConst([0, 1, 2, 0, 0, 1], dtype=torch.long)
_INERT_TRIU_J = _CachedConst([0, 1, 2, 1, 2, 2], dtype=torch.long)
_MJMINVAL = _CachedConst(mujoco.mjMINVAL)


def kinematics(m: Model, d: Data) -> Data:
    """Converts position/velocity from generalized coordinates to maximal."""

    def fn(carry, jnt_typs, jnt_pos, jnt_axis, qpos, qpos0, pos, quat):
        # calculate joint anchors, axes, body pos and quat in global frame
        # also normalize qpos while we're at it

        if carry is not None:
            _, _, _, parent_pos, parent_quat, _ = carry
            pos = parent_pos + math.rotate(pos, parent_quat)
            quat = math.quat_mul(parent_quat, quat)

        anchors, axes = [], []

        qpos_i = 0
        for i, jnt_typ in enumerate(jnt_typs):
            if jnt_typ == JointType.FREE:
                anchor, axis = (
                    qpos[qpos_i : qpos_i + 3],
                    torch.eye(3, dtype=qpos.dtype, device=qpos.device)[2],
                )
            else:
                anchor = math.rotate(jnt_pos[i], quat) + pos
                axis = math.rotate(jnt_axis[i], quat)
            anchors, axes = anchors + [anchor], axes + [axis]

            if jnt_typ == JointType.FREE:
                pos = qpos[qpos_i : qpos_i + 3]
                quat = math.normalize(qpos[qpos_i + 3 : qpos_i + 7])
                qpos = qpos.clone()
                qpos[qpos_i + 3 : qpos_i + 7] = quat
                qpos_i += 7
            elif jnt_typ == JointType.BALL:
                qloc = math.normalize(qpos[qpos_i : qpos_i + 4])
                qpos = qpos.clone()
                qpos[qpos_i : qpos_i + 4] = qloc
                quat = math.quat_mul(quat, qloc)
                pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
                qpos_i += 4
            elif jnt_typ == JointType.HINGE:
                angle = qpos[qpos_i] - qpos0[qpos_i]
                qloc = math.axis_angle_to_quat(jnt_axis[i], angle)
                quat = math.quat_mul(quat, qloc)
                pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
                qpos_i += 1
            elif jnt_typ == JointType.SLIDE:
                pos = pos + axis * (qpos[qpos_i] - qpos0[qpos_i])
                qpos_i += 1
            else:
                raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

        anchor = torch.stack(anchors) if anchors else torch.empty((0, 3), dtype=qpos.dtype, device=qpos.device)
        axis = torch.stack(axes) if axes else torch.empty((0, 3), dtype=qpos.dtype, device=qpos.device)
        mat = math.quat_to_mat(quat)

        return qpos, anchor, axis, pos, quat, mat

    qpos, xanchor, xaxis, xpos, xquat, xmat = scan.body_tree(
        m,
        fn,
        "jjjqqbb",
        "qjjbbb",
        m.jnt_type,
        m.jnt_pos,
        m.jnt_axis,
        d.qpos,
        m.qpos0,
        m.body_pos,
        m.body_quat,
    )

    if m.nmocap:
        mocap_mask = m.body_mocapid >= 0
        xpos = xpos.clone()
        xpos[mocap_mask] = d.mocap_pos
        mocap_quat = torch.vmap(math.normalize)(d.mocap_quat)
        xquat = xquat.clone()
        xquat[mocap_mask] = mocap_quat
        xmat = xmat.clone()
        xmat[mocap_mask] = torch.vmap(math.quat_to_mat)(mocap_quat)

    v_local_to_global = torch.vmap(support.local_to_global)

    xipos, ximat = v_local_to_global(xpos, xquat, m.body_ipos, m.body_iquat)
    kwargs = dict(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos,
                  xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)

    if m.ngeom:
        geom_xpos, geom_xmat = v_local_to_global(xpos[m.geom_bodyid_t], xquat[m.geom_bodyid_t], m.geom_pos, m.geom_quat)
        kwargs.update(geom_xpos=geom_xpos, geom_xmat=geom_xmat)

    if m.nsite:
        site_xpos, site_xmat = v_local_to_global(xpos[m.site_bodyid_t], xquat[m.site_bodyid_t], m.site_pos, m.site_quat)
        kwargs.update(site_xpos=site_xpos, site_xmat=site_xmat)

    d.update_(**kwargs)
    return d


def com_pos(m: Model, d: Data) -> Data:
    """Maps inertias and motion dofs to global frame centered at subtree-CoM."""

    # calculate center of mass of each subtree
    def subtree_sum(carry, xipos, body_mass):
        pos, mass = xipos * body_mass, body_mass
        if carry is not None:
            subtree_pos, subtree_mass = carry
            pos, mass = pos + subtree_pos, mass + subtree_mass
        return pos, mass

    pos, mass = scan.body_tree(m, subtree_sum, "bb", "bb", d.xipos, m.body_mass, reverse=True)
    cond = torch.tile(mass < mujoco.mjMINVAL, (3, 1)).T
    subtree_com = torch.where(
        cond,
        d.xipos,
        torch.vmap(torch.divide)(pos, torch.maximum(mass, _MJMINVAL.get(mass.dtype, mass.device))),
    )

    # map inertias to frame centered at subtree_com
    @torch.vmap
    def inert_com(inert, ximat, off, mass):
        h = math.cross(off.unsqueeze(0).expand(3, 3), -torch.eye(3, dtype=off.dtype, device=off.device))
        inert = math.matmul_unroll(ximat * inert, ximat.T)
        inert = inert + math.matmul_unroll(h, h.T) * mass
        # cinert is triu(inert), mass * off, mass
        inert = inert[(_INERT_TRIU_I.get(torch.long, inert.device), _INERT_TRIU_J.get(torch.long, inert.device))]
        return torch.cat([inert, off * mass, mass.unsqueeze(0)])

    root_com = subtree_com[m.body_rootid_t]
    offset = d.xipos - root_com
    cinert = inert_com(m.body_inertia, d.ximat, offset, m.body_mass)

    # map motion dofs to global frame centered at subtree_com
    def cdof_fn(jnt_typs, root_com, xmat, xanchor, xaxis):
        cdofs = []

        dof_com_fn = lambda a, o: torch.cat([a, math.cross(a, o)])

        for i, jnt_typ in enumerate(jnt_typs):
            offset = root_com - xanchor[i]
            if jnt_typ == JointType.FREE:
                cdofs.append(torch.eye(6, dtype=xaxis.dtype, device=xaxis.device)[3:])  # free translation
                cdofs.append(torch.vmap(dof_com_fn, (0, None))(xmat.T, offset))
            elif jnt_typ == JointType.BALL:
                cdofs.append(torch.vmap(dof_com_fn, (0, None))(xmat.T, offset))
            elif jnt_typ == JointType.HINGE:
                cdof = dof_com_fn(xaxis[i], offset)
                cdofs.append(cdof.unsqueeze(0))
            elif jnt_typ == JointType.SLIDE:
                cdof = torch.cat([torch.zeros((3,), dtype=xaxis.dtype, device=xaxis.device), xaxis[i]])
                cdofs.append(cdof.unsqueeze(0))
            else:
                raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

        cdof = torch.cat(cdofs) if cdofs else torch.empty((0, 6), dtype=xaxis.dtype, device=xaxis.device)

        return cdof

    cdof = scan.flat(
        m,
        cdof_fn,
        "jbbjj",
        "v",
        m.jnt_type,
        root_com,
        d.xmat,
        d.xanchor,
        d.xaxis,
    )

    d.update_(subtree_com=subtree_com, cinert=cinert, cdof=cdof)
    return d


def crb(m: Model, d: Data) -> Data:
    """Runs composite rigid body inertia algorithm."""

    def crb_fn(crb_child, crb_body):
        if crb_child is not None:
            crb_body = crb_body + crb_child
        return crb_body

    crb_body = scan.body_tree(m, crb_fn, "b", "b", d.cinert, reverse=True)
    crb_body = crb_body.clone()
    crb_body[0] = 0.0

    crb_dof = crb_body[m.dof_bodyid_t]
    crb_cdof = torch.vmap(math.inert_mul)(crb_dof, d.cdof)
    qm = support.make_m(m, crb_cdof, d.cdof, m.dof_armature)

    d.update_(crb=crb_body, qM=qm)
    return d


def factor_m(m: Model, d: Data) -> Data:
    """Gets sparse L'*D*L factorization of inertia-like matrix M, assumed spd."""

    if not support.is_sparse(m):
        L = torch.linalg.cholesky(d.qM)
        d.update_(qLD=L)
        return d

    qld = d.qM.clone()

    for rows, madr_ijs, pivots, out in m._device_precomp["factor_m_updates"]:
        qld_update = -(qld[madr_ijs] / qld[pivots]) * qld[rows]
        qld = qld.clone()
        qld[out] = qld[out] + qld_update

    qld_diag = qld[m.dof_Madr_t][:]
    qld = qld / qld[m.factor_m_madr_ds_t]
    qld = qld.clone()
    qld[m.dof_Madr_t] = qld_diag

    d.update_(qLD=qld, qLDiagInv=1 / qld_diag)
    return d


def solve_m(m: Model, d: Data, x: torch.Tensor) -> torch.Tensor:
    """Computes sparse backsubstitution:  x = inv(L'*D*L)*y ."""

    if not support.is_sparse(m):
        return torch.cholesky_solve(x.unsqueeze(-1), d.qLD).squeeze(-1)

    # x <- inv(L') * x
    for j_t, madr_t, i_t in m._device_precomp["solve_m_updates_j"]:
        x = x.clone()
        x[j_t] = x[j_t] + (-d.qLD[madr_t] * x[i_t])

    # x <- inv(D) * x
    x = x * d.qLDiagInv

    # x <- inv(L) * x
    for i_t, madr_t, j_t in m._device_precomp["solve_m_updates_i"]:
        x = x.clone()
        x[i_t] = x[i_t] + (-d.qLD[madr_t] * x[j_t])

    return x


def dense_m(m: Model, d: Data) -> torch.Tensor:
    """Reconstitute dense mass matrix from qM."""

    if not support.is_sparse(m):
        return d.qM

    mat = torch.zeros((m.nv, m.nv), dtype=d.qM.dtype, device=d.qM.device)
    mat[(m.sparse_i_t, m.sparse_j_t)] = d.qM[m.sparse_madr_t]
    mat = torch.diag(d.qM[m.dof_Madr_t]) + mat + mat.T

    return mat


def mul_m(m: Model, d: Data, vec: torch.Tensor) -> torch.Tensor:
    """Multiply vector by inertia matrix."""

    if not support.is_sparse(m):
        return d.qM @ vec

    diag_mul = d.qM[m.dof_Madr_t] * vec

    out = diag_mul.clone()
    out = out.index_add(0, m.sparse_i_t, d.qM[m.sparse_madr_t] * vec[m.sparse_j_t])
    out = out.index_add(0, m.sparse_j_t, d.qM[m.sparse_madr_t] * vec[m.sparse_i_t])

    return out


def com_vel(m: Model, d: Data) -> Data:
    """Computes cvel, cdof_dot."""

    def fn(parent, jnt_typs, cdof, qvel):
        cvel = torch.zeros((6,), dtype=cdof.dtype, device=cdof.device) if parent is None else parent[0]

        cross_fn = lambda cvel_, cdof_: (
            torch.vmap(math.motion_cross, (None, 0))(cvel_, cdof_) if cdof_.shape[0] > 0 else cdof_
        )
        cdof_x_qvel = cdof * qvel.unsqueeze(-1)

        dof_beg = 0
        cdof_dots = []
        for jnt_typ in jnt_typs:
            dof_end = dof_beg + JointType(jnt_typ).dof_width()
            if jnt_typ == JointType.FREE:
                cvel = cvel + torch.sum(cdof_x_qvel[:3], dim=0)
                cdof_ang_dot = cross_fn(cvel, cdof[3:])
                cvel = cvel + torch.sum(cdof_x_qvel[3:], dim=0)
                cdof_dots.append(torch.cat((torch.zeros((3, 6), dtype=cdof.dtype, device=cdof.device), cdof_ang_dot)))
            else:
                cdof_dots.append(cross_fn(cvel, cdof[dof_beg:dof_end]))
                cvel = cvel + torch.sum(cdof_x_qvel[dof_beg:dof_end], dim=0)
            dof_beg = dof_end

        cdof_dot = torch.cat(cdof_dots) if cdof_dots else torch.empty((0, 6))
        return cvel, cdof_dot

    cvel, cdof_dot = scan.body_tree(
        m,
        fn,
        "jvv",
        "bv",
        m.jnt_type,
        d.cdof,
        d.qvel,
    )

    d.update_(cvel=cvel, cdof_dot=cdof_dot)
    return d


def rne(m: Model, d: Data, flg_acc: bool = False) -> Data:
    """Computes inverse dynamics using the recursive Newton-Euler algorithm."""

    # forward scan over tree: accumulate link center of mass acceleration
    def cacc_fn(cacc, cdof_dot, qvel, cdof, qacc):
        if cacc is None:
            if m.opt.disableflags & DisableBit.GRAVITY:
                cacc = torch.zeros((6,), dtype=cdof_dot.dtype, device=cdof_dot.device)
            else:
                cacc = torch.cat((torch.zeros((3,), dtype=cdof_dot.dtype, device=cdof_dot.device), -m.opt.gravity))

        vm = cdof_dot * qvel.unsqueeze(-1)
        vm_sum = torch.sum(vm, dim=0)
        cacc = cacc + vm_sum

        if flg_acc:
            cacc = cacc + torch.sum(cdof * qacc.unsqueeze(-1), dim=0)

        return cacc

    cacc = scan.body_tree(m, cacc_fn, "vvvv", "b", d.cdof_dot, d.qvel, d.cdof, d.qacc)

    def frc(cinert, cacc, cvel):
        frc = math.inert_mul(cinert, cacc)
        frc = frc + math.motion_cross_force(cvel, math.inert_mul(cinert, cvel))

        return frc

    loc_cfrc = torch.vmap(frc)(d.cinert, cacc, d.cvel)

    # backward scan up tree: accumulate body forces
    def cfrc_fn(cfrc_child, cfrc):
        if cfrc_child is not None:
            cfrc = cfrc + cfrc_child
        return cfrc

    cfrc = scan.body_tree(m, cfrc_fn, "b", "b", loc_cfrc, reverse=True)
    qfrc_bias = torch.vmap(torch.dot)(d.cdof, cfrc[m.dof_bodyid_t])

    d.update_(qfrc_bias=qfrc_bias)
    return d


def tendon(m: Model, d: Data) -> Data:
    """Computes tendon lengths and moments (Jacobian)."""
    if not m.ntendon:
        return d

    # process joint (fixed) tendons
    (wrap_id_jnt,) = np.nonzero(m.wrap_type == WrapType.JOINT)

    if wrap_id_jnt.size == 0:
        d.update_(
            ten_length=torch.zeros(m.ntendon, dtype=d.qpos.dtype, device=d.qpos.device),
            ten_J=torch.zeros((m.ntendon, m.nv), dtype=d.qpos.dtype, device=d.qpos.device),
        )
        return d

    (tendon_id_jnt,) = np.nonzero(np.isin(m.tendon_adr, wrap_id_jnt))

    ntendon_jnt = tendon_id_jnt.size
    wrap_objid_jnt = m.wrap_objid[wrap_id_jnt]
    tendon_num_jnt = m.tendon_num[tendon_id_jnt]

    # moment_jnt[i] = wrap_prm (coefficient) for each wrap element
    moment_jnt = torch.as_tensor(m.wrap_prm[wrap_id_jnt], device=d.qpos.device).to(dtype=d.qpos.dtype)
    qpos_vals = d.qpos[m.jnt_qposadr[wrap_objid_jnt]]

    # tendon length = sum of (coefficient * joint position) per tendon
    segment_ids = torch.arange(ntendon_jnt, device=d.qpos.device).repeat_interleave(
        torch.as_tensor(tendon_num_jnt, device=d.qpos.device)
    )
    ten_length = torch.zeros(m.ntendon, dtype=d.qpos.dtype, device=d.qpos.device)
    ten_length_jnt = torch.zeros(ntendon_jnt, dtype=d.qpos.dtype, device=d.qpos.device)
    ten_length_jnt = ten_length_jnt.index_add(0, segment_ids, moment_jnt * qpos_vals)
    ten_length[tendon_id_jnt] = ten_length_jnt

    # tendon Jacobian: ten_J[tendon_id, dof_adr] = wrap_prm (coefficient)
    ten_J = torch.zeros((m.ntendon, m.nv), dtype=d.qpos.dtype, device=d.qpos.device)
    adr_moment_jnt = torch.as_tensor(tendon_id_jnt, device=d.qpos.device).repeat_interleave(
        torch.as_tensor(tendon_num_jnt, device=d.qpos.device)
    )
    dofadr_moment_jnt = m.jnt_dofadr[wrap_objid_jnt]
    ten_J[adr_moment_jnt, dofadr_moment_jnt] = moment_jnt

    d.update_(ten_length=ten_length, ten_J=ten_J)
    return d


def tendon_armature(m: Model, d: Data) -> Data:
    """Add tendon armature to qM."""
    if not m.ntendon:
        return d

    # JTAJ = J^T @ diag(armature) @ J
    JTAJ = d.ten_J.T @ (d.ten_J * m.tendon_armature.unsqueeze(1))

    if support.is_sparse(m):
        ij = []
        for i in range(m.nv):
            j = i
            while j > -1:
                ij.append((i, j))
                j = m.dof_parentid[j]
        i_idx, j_idx = zip(*ij)
        i_idx = torch.tensor(i_idx, dtype=torch.long, device=d.qM.device)
        j_idx = torch.tensor(j_idx, dtype=torch.long, device=d.qM.device)
        JTAJ = JTAJ[(i_idx, j_idx)]

    d.update_(qM=d.qM + JTAJ)
    return d


def _moment_row(values: torch.Tensor, dofadr: torch.Tensor, nv: int) -> torch.Tensor:
    """Build a ``(nv,)`` moment row with ``values`` scattered at ``dofadr``.

    All operations are out-of-place and use tensor indices so the function
    is compatible with both ``torch.vmap`` and ``torch.compile``.
    """
    k = values.shape[-1]
    idx = torch.arange(k, dtype=torch.long, device=values.device) + dofadr
    return torch.zeros(nv, dtype=values.dtype, device=values.device).scatter(0, idx, values)


def transmission(m: Model, d: Data) -> Data:
    """Computes actuator/transmission lengths and moments.

    All per-actuator results are built out-of-place (list + stack) so that
    the function is compatible with ``torch.vmap``.
    """
    if not m.nu:
        return d

    dtype = d.qpos.dtype
    device = d.qpos.device
    zero = torch.zeros((), dtype=dtype, device=device)
    lengths: list[torch.Tensor] = []
    moments: list[torch.Tensor] = []

    for i in range(m.nu):
        trntype, trnid, jnt_type_i, dofadr, qposadr = m.actuator_info[i]
        gear = m.actuator_gear[i]

        if trntype == TrnType.TENDON:
            lengths.append(d.ten_length[trnid] * gear[0])
            moments.append(d.ten_J[trnid] * gear[0])
            continue

        jnt_typ = JointType(jnt_type_i)

        if jnt_typ == JointType.FREE:
            qpos = d.qpos[qposadr : qposadr + 7]
            if trntype == TrnType.JOINTINPARENT:
                quat_neg = math.quat_inv(qpos[3:])
                gearaxis = math.rotate(gear[3:], quat_neg)
                values = torch.cat([gear[:3], gearaxis])
            else:
                values = gear[:6]
            lengths.append(zero)
            moments.append(_moment_row(values, dofadr, m.nv))
        elif jnt_typ == JointType.BALL:
            qpos = d.qpos[qposadr : qposadr + 4]
            axis, angle = math.quat_to_axis_angle(qpos)
            gearaxis = gear[:3]
            if trntype == TrnType.JOINTINPARENT:
                quat_neg = math.quat_inv(qpos)
                gearaxis = math.rotate(gear[:3], quat_neg)
            lengths.append(torch.dot(axis * angle, gearaxis))
            moments.append(_moment_row(gearaxis, dofadr, m.nv))
        elif jnt_typ in (JointType.SLIDE, JointType.HINGE):
            lengths.append(d.qpos[qposadr] * gear[0])
            moments.append(_moment_row(gear[0].unsqueeze(0), dofadr, m.nv))
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

    length = torch.stack(lengths)
    moment = torch.stack(moments)
    d.update_(actuator_length=length, actuator_moment=moment)
    return d
