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
import torch

from mujoco_torch._src import math, scan, support
from mujoco_torch._src.math import _CachedConst

# pylint: disable=g-importing-member
from mujoco_torch._src.types import CamLightType, Data, DisableBit, JointType, Model, TrnType

# pylint: enable=g-importing-member

_INERT_TRIU_I = _CachedConst([0, 1, 2, 0, 0, 1], dtype=torch.long)
_INERT_TRIU_J = _CachedConst([0, 1, 2, 1, 2, 2], dtype=torch.long)
_MJMINVAL = _CachedConst(mujoco.mjMINVAL)

_FREE = JointType.FREE
_BALL = JointType.BALL
_SLIDE = JointType.SLIDE
_HINGE = JointType.HINGE


def kinematics(m: Model, d: Data) -> Data:
    """Converts position/velocity from generalized coordinates to maximal."""

    max_j = m.max_joints_per_body
    has_free = _FREE in m.joint_types_present
    has_ball = _BALL in m.joint_types_present
    has_hinge = _HINGE in m.joint_types_present
    has_slide = _SLIDE in m.joint_types_present

    def fn(carry, jnt_typs, jnt_pos, jnt_axis, qpos, qpos0, pos, quat, n_jnts):
        if carry is not None:
            _, _, _, parent_pos, parent_quat, _ = carry
            pos = parent_pos + math.rotate(pos, parent_quat)
            quat = math.quat_mul(parent_quat, quat)

        dtype, device = pos.dtype, pos.device
        anchor_list = []
        axis_list = []
        qpos_out_list = []

        for j in range(max_j):
            jt = jnt_typs[j]
            q = qpos[j]
            q0 = qpos0[j]

            # anchor & axis
            other_anchor = math.rotate(jnt_pos[j], quat) + pos
            other_axis = math.rotate(jnt_axis[j], quat)
            if has_free:
                free_anchor = q[:3]
                free_axis = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
                anchor = torch.where(jt == _FREE, free_anchor, other_anchor)
                axis = torch.where(jt == _FREE, free_axis, other_axis)
            else:
                anchor = other_anchor
                axis = other_axis
            anchor_list.append(anchor)
            axis_list.append(axis)

            # Build pos/quat from bottom up: start with the simplest type present
            if has_hinge:
                hinge_angle = q[0] - q0[0]
                hinge_qloc = math.axis_angle_to_quat(jnt_axis[j], hinge_angle)
                hinge_quat = math.quat_mul(quat, hinge_qloc)
                hinge_pos = anchor - math.rotate(jnt_pos[j], hinge_quat)

            if has_slide:
                slide_pos = pos + axis * (q[0] - q0[0])

            if has_hinge and has_slide:
                new_pos = torch.where(jt == _HINGE, hinge_pos, slide_pos)
                new_quat = torch.where(jt == _HINGE, hinge_quat, quat)
            elif has_hinge:
                new_pos = hinge_pos
                new_quat = hinge_quat
            elif has_slide:
                new_pos = slide_pos
                new_quat = quat
            else:
                new_pos = pos
                new_quat = quat
            new_q_out = q

            if has_ball:
                ball_qloc = math.normalize(q[:4])
                ball_quat = math.quat_mul(quat, ball_qloc)
                ball_pos = anchor - math.rotate(jnt_pos[j], ball_quat)
                ball_q_out = torch.cat([ball_qloc, q[4:]])
                new_pos = torch.where(jt == _BALL, ball_pos, new_pos)
                new_quat = torch.where(jt == _BALL, ball_quat, new_quat)
                new_q_out = torch.where(jt == _BALL, ball_q_out, new_q_out)

            if has_free:
                free_pos = q[:3]
                free_quat = math.normalize(q[3:7])
                free_q_out = torch.cat([q[:3], free_quat])
                new_pos = torch.where(jt == _FREE, free_pos, new_pos)
                new_quat = torch.where(jt == _FREE, free_quat, new_quat)
                new_q_out = torch.where(jt == _FREE, free_q_out, new_q_out)

            qpos_out_list.append(new_q_out)

            is_valid = j < n_jnts
            pos = torch.where(is_valid, new_pos, pos)
            quat = torch.where(is_valid, new_quat, quat)

        anchors = torch.stack(anchor_list)
        axes = torch.stack(axis_list)
        qpos_out = torch.stack(qpos_out_list)
        mat = math.quat_to_mat(quat)
        return qpos_out, anchors, axes, pos, quat, mat

    qpos, xanchor, xaxis, xpos, xquat, xmat = scan.body_tree(
        m,
        fn,
        "jjjqqbbb",
        "qjjbbb",
        m.jnt_type,
        m.jnt_pos,
        m.jnt_axis,
        d.qpos,
        m.qpos0,
        m.body_pos,
        m.body_quat,
        m.n_joints_per_body,
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
    kwargs = dict(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos, xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)

    if m.ngeom:
        geom_xpos, geom_xmat = v_local_to_global(xpos[m.geom_bodyid_t], xquat[m.geom_bodyid_t], m.geom_pos, m.geom_quat)
        kwargs.update(geom_xpos=geom_xpos, geom_xmat=geom_xmat)

    if m.nsite:
        site_xpos, site_xmat = v_local_to_global(xpos[m.site_bodyid_t], xquat[m.site_bodyid_t], m.site_pos, m.site_quat)
        kwargs.update(site_xpos=site_xpos, site_xmat=site_xmat)

    if m.ncam:
        cam_xpos, cam_xmat = v_local_to_global(
            xpos[m.cam_bodyid_t], xquat[m.cam_bodyid_t], m.cam_pos, m.cam_quat,
        )

        for ci in range(m.ncam):
            mode = int(m.cam_mode[ci])
            if mode == int(CamLightType.FIXED):
                pass
            elif mode == int(CamLightType.TRACK):
                cam_xpos = cam_xpos.clone()
                cam_xpos[ci] = xpos[int(m.cam_bodyid[ci])] + m.cam_pos0[ci]
                cam_xmat = cam_xmat.clone()
                cam_xmat[ci] = m.cam_mat0[ci].reshape(3, 3)
            elif mode == int(CamLightType.TRACKCOM):
                cam_xpos = cam_xpos.clone()
                if hasattr(d, "subtree_com") and d.subtree_com is not None:
                    cam_xpos[ci] = d.subtree_com[int(m.cam_bodyid[ci])] + math.rotate(
                        m.cam_pos[ci], xquat[int(m.cam_bodyid[ci])]
                    )
            elif mode == int(CamLightType.TARGETBODY):
                target_id = int(m.cam_targetbodyid[ci])
                if target_id >= 0:
                    target_pos = xpos[target_id]
                    fwd = math.normalize(target_pos - cam_xpos[ci])
                    up_hint = torch.tensor([0.0, 0.0, 1.0], dtype=fwd.dtype, device=fwd.device)
                    right = math.normalize(math.cross(fwd, up_hint))
                    up = math.cross(right, fwd)
                    cam_xmat = cam_xmat.clone()
                    cam_xmat[ci] = torch.stack([right, up, -fwd], dim=-1)
            elif mode == int(CamLightType.TARGETBODYCOM):
                target_id = int(m.cam_targetbodyid[ci])
                if target_id >= 0 and hasattr(d, "subtree_com") and d.subtree_com is not None:
                    target_pos = d.subtree_com[target_id]
                    fwd = math.normalize(target_pos - cam_xpos[ci])
                    up_hint = torch.tensor([0.0, 0.0, 1.0], dtype=fwd.dtype, device=fwd.device)
                    right = math.normalize(math.cross(fwd, up_hint))
                    up = math.cross(right, fwd)
                    cam_xmat = cam_xmat.clone()
                    cam_xmat[ci] = torch.stack([right, up, -fwd], dim=-1)

        kwargs.update(cam_xpos=cam_xpos, cam_xmat=cam_xmat)

    if m.nlight:
        body_quat = xquat[m.light_bodyid_t]
        light_xpos = xpos[m.light_bodyid_t] + torch.vmap(math.rotate)(m.light_pos, body_quat)
        light_xdir = torch.vmap(math.rotate)(m.light_dir, body_quat)
        kwargs.update(light_xpos=light_xpos, light_xdir=light_xdir)

    d.update_(**kwargs)
    return d


def com_pos(m: Model, d: Data) -> Data:
    """Maps inertias and motion dofs to global frame centered at subtree-CoM."""

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

    @torch.vmap
    def inert_com(inert, ximat, off, mass):
        h = math.cross(off.unsqueeze(0).expand(3, 3), -torch.eye(3, dtype=off.dtype, device=off.device))
        inert = math.matmul_unroll(ximat * inert, ximat.T)
        inert = inert + math.matmul_unroll(h, h.T) * mass
        inert = inert[(_INERT_TRIU_I.get(torch.long, inert.device), _INERT_TRIU_J.get(torch.long, inert.device))]
        return torch.cat([inert, off * mass, mass.unsqueeze(0)])

    root_com = subtree_com[m.body_rootid_t]
    offset = d.xipos - root_com
    cinert = inert_com(m.body_inertia, d.ximat, offset, m.body_mass)

    max_j = m.max_joints_per_body
    has_free = _FREE in m.joint_types_present
    has_ball = _BALL in m.joint_types_present
    has_hinge = _HINGE in m.joint_types_present
    has_slide = _SLIDE in m.joint_types_present
    max_dw = m.max_dof_per_jnt

    def cdof_fn(jnt_typs, root_com, xmat, xanchor, xaxis):
        dtype, device = xaxis.dtype, xaxis.device
        dof_com_fn = lambda a, o: torch.cat([a, math.cross(a, o)])
        pad_rows = max_dw - 1

        cdof_list = []
        for j in range(max_j):
            jt = jnt_typs[j]
            off = root_com - xanchor[j]

            if has_hinge:
                hinge_val = dof_com_fn(xaxis[j], off)
                hinge_cdof = torch.cat([hinge_val.unsqueeze(0), torch.zeros(pad_rows, 6, dtype=dtype, device=device)])

            if has_slide:
                slide_val = torch.cat([torch.zeros(3, dtype=dtype, device=device), xaxis[j]])
                slide_cdof = torch.cat([slide_val.unsqueeze(0), torch.zeros(pad_rows, 6, dtype=dtype, device=device)])

            if has_hinge and has_slide:
                result = torch.where(jt == _HINGE, hinge_cdof, slide_cdof)
            elif has_hinge:
                result = hinge_cdof
            elif has_slide:
                result = slide_cdof
            else:
                result = torch.zeros(max_dw, 6, dtype=dtype, device=device)

            if has_ball:
                ball_rot = torch.vmap(dof_com_fn, (0, None))(xmat.T, off)
                ball_cdof = torch.cat([ball_rot, torch.zeros(max_dw - 3, 6, dtype=dtype, device=device)])
                result = torch.where(jt == _BALL, ball_cdof, result)

            if has_free:
                free_trans = torch.eye(6, dtype=dtype, device=device)[3:]
                free_rot = torch.vmap(dof_com_fn, (0, None))(xmat.T, off)
                free_cdof = torch.cat([free_trans, free_rot])
                result = torch.where(jt == _FREE, free_cdof, result)

            cdof_list.append(result)

        return torch.stack(cdof_list)

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
        qld = qld.scatter(0, out, qld[out] + qld_update)

    qld_diag = qld[m.dof_Madr_t][:]
    qld = qld / qld[m.factor_m_madr_ds_t]
    qld[m.dof_Madr_t] = qld_diag

    d.update_(qLD=qld, qLDiagInv=1 / qld_diag)
    return d


def solve_m(m: Model, d: Data, x: torch.Tensor) -> torch.Tensor:
    """Computes sparse backsubstitution:  x = inv(L'*D*L)*y ."""

    if not support.is_sparse(m):
        return torch.cholesky_solve(x.unsqueeze(-1), d.qLD).squeeze(-1)

    for j_t, madr_t, i_t in m._device_precomp["solve_m_updates_j"]:
        update = x[j_t] + (-d.qLD[madr_t] * x[i_t])
        x = x.scatter(0, j_t, update)

    x = x * d.qLDiagInv

    for i_t, madr_t, j_t in m._device_precomp["solve_m_updates_i"]:
        update = x[i_t] + (-d.qLD[madr_t] * x[j_t])
        x = x.scatter(0, i_t, update)

    return x


def dense_m(m: Model, d: Data) -> torch.Tensor:
    if not support.is_sparse(m):
        return d.qM
    mat = torch.zeros((m.nv, m.nv), dtype=d.qM.dtype, device=d.qM.device)
    mat[(m.sparse_i_t, m.sparse_j_t)] = d.qM[m.sparse_madr_t]
    mat = torch.diag(d.qM[m.dof_Madr_t]) + mat + mat.T
    return mat


def mul_m(m: Model, d: Data, vec: torch.Tensor) -> torch.Tensor:
    if not support.is_sparse(m):
        return d.qM @ vec
    diag_mul = d.qM[m.dof_Madr_t] * vec
    out = diag_mul.clone()
    out = out.index_add(0, m.sparse_i_t, d.qM[m.sparse_madr_t] * vec[m.sparse_j_t])
    out = out.index_add(0, m.sparse_j_t, d.qM[m.sparse_madr_t] * vec[m.sparse_i_t])
    return out


def com_vel(m: Model, d: Data) -> Data:
    """Computes cvel, cdof_dot."""

    max_j = m.max_joints_per_body
    has_free = _FREE in m.joint_types_present
    has_ball = _BALL in m.joint_types_present
    max_dw = m.max_dof_per_jnt

    def fn(parent, jnt_typs, cdof, qvel, n_jnts):
        dtype, device = cdof.dtype, cdof.device
        cvel = torch.zeros((6,), dtype=dtype, device=device) if parent is None else parent[0]

        cdof_x_qvel = cdof * qvel.unsqueeze(-1)  # (max_j, max_dw, 6)

        cross_fn = lambda cvel_, cdof_row: math.motion_cross(cvel_, cdof_row)

        cdof_dot_list = []
        for j in range(max_j):
            jt = jnt_typs[j]
            cd = cdof[j]     # (max_dw, 6)
            cxq = cdof_x_qvel[j]  # (max_dw, 6)

            # Non-FREE path with DOF validity mask
            if has_ball:
                dw = torch.where(jt == _BALL, 3, 1)
                dof_mask = (torch.arange(max_dw, device=device) < dw).to(dtype=dtype)
                masked_cxq = cxq * dof_mask.unsqueeze(-1)
            else:
                masked_cxq = cxq[:1]

            other_cdof_dot = torch.vmap(cross_fn, (None, 0))(cvel, cd)
            new_cvel = cvel + torch.sum(masked_cxq, dim=0)
            new_cdof_dot = other_cdof_dot

            if has_free:
                free_cvel_after_trans = cvel + torch.sum(cxq[:3], dim=0)
                free_cdof_ang_dot = torch.vmap(cross_fn, (None, 0))(free_cvel_after_trans, cd[3:6])
                free_cvel = free_cvel_after_trans + torch.sum(cxq[3:6], dim=0)
                free_cdof_dot = torch.cat([
                    torch.zeros(3, 6, dtype=dtype, device=device),
                    free_cdof_ang_dot,
                ])
                new_cdof_dot = torch.where(jt == _FREE, free_cdof_dot, new_cdof_dot)
                new_cvel = torch.where(jt == _FREE, free_cvel, new_cvel)

            is_valid = j < n_jnts
            cvel = torch.where(is_valid, new_cvel, cvel)
            cdof_dot_list.append(
                torch.where(is_valid, new_cdof_dot, torch.zeros_like(new_cdof_dot))
            )

        cdof_dots = torch.stack(cdof_dot_list)
        return cvel, cdof_dots

    cvel, cdof_dot = scan.body_tree(
        m,
        fn,
        "jvvb",
        "bv",
        m.jnt_type,
        d.cdof,
        d.qvel,
        m.n_joints_per_body,
    )

    d.update_(cvel=cvel, cdof_dot=cdof_dot)
    return d


def rne(m: Model, d: Data, flg_acc: bool = False) -> Data:
    """Computes inverse dynamics using the recursive Newton-Euler algorithm."""

    max_j = m.max_joints_per_body
    has_free = _FREE in m.joint_types_present
    has_ball = _BALL in m.joint_types_present
    max_dw = m.max_dof_per_jnt

    def cacc_fn(cacc, cdof_dot, qvel, cdof, qacc, jnt_typs, n_jnts):
        device = cdof_dot.device
        dtype = cdof_dot.dtype

        if cacc is None:
            if m.opt.disableflags & DisableBit.GRAVITY:
                cacc = torch.zeros((6,), dtype=dtype, device=device)
            else:
                cacc = torch.cat((torch.zeros((3,), dtype=dtype, device=device), -m.opt.gravity))

        # Per-DOF validity mask: (max_j, max_dw)
        if has_free or has_ball:
            if has_free and has_ball:
                dw = torch.where(
                    jnt_typs.unsqueeze(-1) == _FREE, 6,
                    torch.where(jnt_typs.unsqueeze(-1) == _BALL, 3, 1),
                )
            elif has_free:
                dw = torch.where(jnt_typs.unsqueeze(-1) == _FREE, 6, 1)
            else:
                dw = torch.where(jnt_typs.unsqueeze(-1) == _BALL, 3, 1)

            dof_idx = torch.arange(max_dw, device=device).unsqueeze(0)
            dof_valid = (dof_idx < dw)
            jnt_valid = (torch.arange(max_j, device=device).unsqueeze(-1) < n_jnts)
            mask = (dof_valid & jnt_valid).to(dtype=dtype)
        else:
            jnt_valid = (torch.arange(max_j, device=device).unsqueeze(-1) < n_jnts)
            mask = jnt_valid.to(dtype=dtype)

        vm = cdof_dot * qvel.unsqueeze(-1)
        vm = vm * mask.unsqueeze(-1)
        cacc = cacc + torch.sum(vm, dim=(0, 1))

        if flg_acc:
            am = cdof * qacc.unsqueeze(-1)
            am = am * mask.unsqueeze(-1)
            cacc = cacc + torch.sum(am, dim=(0, 1))

        return cacc

    cacc = scan.body_tree(
        m, cacc_fn, "vvvvjb", "b",
        d.cdof_dot, d.qvel, d.cdof, d.qacc, m.jnt_type, m.n_joints_per_body,
    )

    def frc(cinert, cacc, cvel):
        frc = math.inert_mul(cinert, cacc)
        frc = frc + math.motion_cross_force(cvel, math.inert_mul(cinert, cvel))
        return frc

    loc_cfrc = torch.vmap(frc)(d.cinert, cacc, d.cvel)

    def cfrc_fn(cfrc_child, cfrc):
        if cfrc_child is not None:
            cfrc = cfrc + cfrc_child
        return cfrc

    cfrc = scan.body_tree(m, cfrc_fn, "b", "b", loc_cfrc, reverse=True)
    qfrc_bias = torch.vmap(torch.dot)(d.cdof, cfrc[m.dof_bodyid_t])

    d.update_(qfrc_bias=qfrc_bias)
    return d


def tendon(m: Model, d: Data) -> Data:
    if not m.ntendon:
        return d
    if not m.tendon_has_jnt:
        d.update_(
            ten_length=torch.zeros(m.ntendon, dtype=d.qpos.dtype, device=d.qpos.device),
            ten_J=torch.zeros((m.ntendon, m.nv), dtype=d.qpos.dtype, device=d.qpos.device),
        )
        return d
    moment_jnt = m.tendon_moment_jnt.data.to(dtype=d.qpos.dtype)
    qpos_vals = d.qpos[m.tendon_qposadr_jnt.data]
    ntendon_jnt = m.tendon_ntendon_jnt
    ten_length = torch.zeros(m.ntendon, dtype=d.qpos.dtype, device=d.qpos.device)
    ten_length_jnt = torch.zeros(ntendon_jnt, dtype=d.qpos.dtype, device=d.qpos.device)
    ten_length_jnt = ten_length_jnt.index_add(0, m.tendon_segment_ids.data, moment_jnt * qpos_vals)
    ten_length[m.tendon_tendon_id_jnt.data] = ten_length_jnt
    ten_J = torch.zeros((m.ntendon, m.nv), dtype=d.qpos.dtype, device=d.qpos.device)
    ten_J[m.tendon_adr_moment_jnt.data, m.tendon_dofadr_moment_jnt.data] = moment_jnt
    d.update_(ten_length=ten_length, ten_J=ten_J)
    return d


def tendon_armature(m: Model, d: Data) -> Data:
    if not m.ntendon:
        return d
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
    k = values.shape[-1]
    idx = torch.arange(k, dtype=torch.long, device=values.device) + dofadr
    return torch.zeros(nv, dtype=values.dtype, device=values.device).scatter(0, idx, values)


def transmission(m: Model, d: Data) -> Data:
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
