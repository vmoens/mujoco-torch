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
"""Core non-smooth constraint functions."""

import mujoco

# pylint: enable=g-importing-member
import torch

from mujoco_torch._src import collision_driver, math, support
from mujoco_torch._src.math import _CachedConst

_MJMINVAL = _CachedConst(mujoco.mjMINVAL)
_ONE = _CachedConst(1.0)
_ZERO_I32 = _CachedConst(0, dtype=torch.int32)


def _vmap_index(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Vmap-compatible indexing: tensor[idx] for scalar idx inside vmap."""
    idx = idx.long().reshape(1)
    ndim = tensor.ndim
    if ndim == 1:
        return tensor.gather(0, idx).squeeze(0)
    elif ndim == 2:
        return tensor.gather(0, idx.unsqueeze(-1).expand(1, tensor.shape[1])).squeeze(0)
    elif ndim == 3:
        return tensor.gather(0, idx.reshape(1, 1, 1).expand(1, tensor.shape[1], tensor.shape[2])).squeeze(0)
    return tensor[idx.squeeze(0)]


# pylint: disable=g-importing-member
from torch.utils._pytree import tree_map

from mujoco_torch._src.dataclasses import MjTensorClass
from mujoco_torch._src.types import Contact, Data, Model


class _Efc(MjTensorClass):
    """Support data for creating constraint matrices."""

    J: torch.Tensor
    pos: torch.Tensor
    pos_norm: torch.Tensor
    invweight: torch.Tensor
    solref: torch.Tensor
    solimp: torch.Tensor
    frictionloss: torch.Tensor


def _kbi(
    m: Model,
    solref: torch.Tensor,
    solimp: torch.Tensor,
    pos: torch.Tensor,
    refsafe: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculates stiffness, damping, and impedance of a constraint."""
    timeconst, dampratio = solref

    if not refsafe:
        timeconst = torch.maximum(timeconst, 2 * m.opt.timestep) * (timeconst > 0)

    dmin, dmax, width, mid, power = solimp

    dmin = torch.clamp(dmin, mujoco.mjMINIMP, mujoco.mjMAXIMP)
    dmax = torch.clamp(dmax, mujoco.mjMINIMP, mujoco.mjMAXIMP)
    width = torch.maximum(_MJMINVAL.get(width.dtype, width.device), width)
    mid = torch.clamp(mid, mujoco.mjMINIMP, mujoco.mjMAXIMP)
    power = torch.maximum(_ONE.get(power.dtype, power.device), power)

    # See https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters
    k = 1 / (dmax * dmax * timeconst * timeconst * dampratio * dampratio)
    b = 2 / (dmax * timeconst)
    # TODO(robotics-simulation): check various solparam settings in model gen test
    k = torch.where(dampratio <= 0, -dampratio / (dmax * dmax), k)
    b = torch.where(timeconst <= 0, -timeconst / dmax, b)

    imp_x = torch.abs(pos) / width
    imp_a = (1.0 / torch.pow(mid, power - 1)) * torch.pow(imp_x, power)
    imp_b = 1 - (1.0 / torch.pow(1 - mid, power - 1)) * torch.pow(1 - imp_x, power)
    imp_y = torch.where(imp_x < mid, imp_a, imp_b)
    imp = dmin + imp_y * (dmax - dmin)
    imp = torch.clamp(imp, dmin, dmax)
    imp = torch.where(imp_x > 1.0, dmax, imp)

    return k, b, imp  # corresponds to K, B, I of efc_KBIP


def _instantiate_equality_connect(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for connect equality constraints."""
    _dev = d.qpos.device
    ids = precomp["ids"].to(_dev)
    id1_t = precomp["id1"].to(_dev)
    id2_t = precomp["id2"].to(_dev)

    data = m.eq_data[ids]
    active = d.eq_active[ids]

    @torch.vmap
    def fn(data, id1, id2, active):
        anchor1, anchor2 = data[0:3], data[3:6]
        pos1 = _vmap_index(d.xmat, id1) @ anchor1 + _vmap_index(d.xpos, id1)
        pos2 = _vmap_index(d.xmat, id2) @ anchor2 + _vmap_index(d.xpos, id2)

        cpos = pos1 - pos2

        jacp1, _ = support.jac(m, d, pos1, id1)
        jacp2, _ = support.jac(m, d, pos2, id2)
        j = (jacp1 - jacp2).T

        result = (j, cpos, math.norm(cpos).unsqueeze(0).expand(3))
        return tree_map(lambda x: x * active, result)

    j, pos, pos_norm = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), fn(data, id1_t, id2_t, active))
    invweight = m.body_invweight0[id1_t, 0] + m.body_invweight0[id2_t, 0]
    invweight = invweight.repeat_interleave(3)
    solref = torch.tile(m.eq_solref[ids], (3, 1))
    solimp = torch.tile(m.eq_solimp[ids], (3, 1))
    frictionloss = torch.zeros_like(pos_norm)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos_norm,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_equality_weld(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for weld equality constraints."""
    _dev = d.qpos.device
    ids = precomp["ids"].to(_dev)
    id1_t = precomp["id1"].to(_dev)
    id2_t = precomp["id2"].to(_dev)

    data = m.eq_data[ids]
    active = d.eq_active[ids]

    @torch.vmap
    def fn(data, id1, id2, active):
        anchor1, anchor2 = data[0:3], data[3:6]
        relpose, torquescale = data[6:10], data[10]

        pos1 = _vmap_index(d.xmat, id1) @ anchor2 + _vmap_index(d.xpos, id1)
        pos2 = _vmap_index(d.xmat, id2) @ anchor1 + _vmap_index(d.xpos, id2)

        cpos = pos1 - pos2

        jacp1, jacr1 = support.jac(m, d, pos1, id1)
        jacp2, jacr2 = support.jac(m, d, pos2, id2)
        jacdifp = jacp1 - jacp2
        jacdifr = (jacr1 - jacr2) * torquescale

        quat = math.quat_mul(_vmap_index(d.xquat, id1), relpose)
        quat1 = math.quat_inv(_vmap_index(d.xquat, id2))
        crot = math.quat_mul(quat1, quat)[1:]

        jac_fn = lambda j: math.quat_mul(math.quat_mul_axis(quat1, j), quat)[1:]
        jacdifr = 0.5 * torch.vmap(jac_fn)(jacdifr)

        j = torch.cat((jacdifp.T, jacdifr.T))
        pos = torch.cat((cpos, crot * torquescale))

        result = (j, pos, math.norm(pos).unsqueeze(0).expand(6))
        return tree_map(lambda x: x * active, result)

    j, pos, pos_norm = tree_map(lambda x: x.reshape(-1, *x.shape[2:]), fn(data, id1_t, id2_t, active))
    invweight = m.body_invweight0[id1_t] + m.body_invweight0[id2_t]
    invweight = invweight.repeat_interleave(3)
    solref = torch.tile(m.eq_solref[ids], (6, 1))
    solimp = torch.tile(m.eq_solimp[ids], (6, 1))
    frictionloss = torch.zeros_like(pos_norm)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos_norm,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_friction(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for DOF and tendon frictionloss."""
    _dev = d.qpos.device
    dof_ids = precomp["dof_ids"].to(_dev)
    tendon_ids = precomp["tendon_ids"].to(_dev)
    size = precomp["size"]

    eye = torch.eye(m.nv, dtype=m.dof_frictionloss.dtype, device=m.dof_frictionloss.device)

    j_dof = eye[dof_ids]
    fl_dof = m.dof_frictionloss[dof_ids]
    iw_dof = m.dof_invweight0[dof_ids]
    sr_dof = m.dof_solref[dof_ids]
    si_dof = m.dof_solimp[dof_ids]

    j_ten = d.ten_J[tendon_ids]
    fl_ten = m.tendon_frictionloss[tendon_ids]
    iw_ten = m.tendon_invweight0[tendon_ids]
    sr_ten = m.tendon_solref_fri[tendon_ids]
    si_ten = m.tendon_solimp_fri[tendon_ids]

    j = torch.cat([j_dof, j_ten])
    frictionloss = torch.cat([fl_dof, fl_ten])
    invweight = torch.cat([iw_dof, iw_ten])
    solref = torch.cat([sr_dof, sr_ten])
    solimp = torch.cat([si_dof, si_ten])
    pos = torch.zeros(size, dtype=frictionloss.dtype, device=frictionloss.device)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[size],
    )


def _instantiate_equality_joint(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for joint equality constraints."""
    _dev = d.qpos.device
    ids = precomp["ids"].to(_dev)
    _id1_t = precomp["id1"].to(_dev)
    id2_t = precomp["id2"].to(_dev)
    dofadr1_t = precomp["dofadr1"].to(_dev)
    dofadr2_t = precomp["dofadr2"].to(_dev)
    qposadr1_t = precomp["qposadr1"].to(_dev)
    qposadr2_t = precomp["qposadr2"].to(_dev)

    data = m.eq_data[ids]
    active = d.eq_active[ids]

    @torch.vmap
    def fn(data, id2, dofadr1, dofadr2, qposadr1, qposadr2, active):
        pos1 = d.qpos.gather(0, qposadr1.unsqueeze(0)).squeeze(0)
        pos2 = d.qpos.gather(0, qposadr2.unsqueeze(0)).squeeze(0)
        ref1 = m.qpos0.gather(0, qposadr1.unsqueeze(0)).squeeze(0)
        ref2 = m.qpos0.gather(0, qposadr2.unsqueeze(0)).squeeze(0)
        pos2, ref2 = pos2 * (id2 > -1), ref2 * (id2 > -1)

        dif = pos2 - ref2
        dif_power = torch.pow(dif, torch.arange(0, 5, dtype=data.dtype, device=data.device))

        deriv = torch.dot(data[1:5], dif_power[:4] * torch.arange(1, 5, dtype=data.dtype, device=data.device))
        j = torch.zeros(m.nv, dtype=data.dtype, device=data.device)
        j = j.scatter(0, dofadr1.unsqueeze(0), torch.ones(1, dtype=data.dtype, device=data.device))
        j = j.scatter(0, dofadr2.unsqueeze(0), (-deriv).unsqueeze(0))
        pos = pos1 - ref1 - torch.dot(data[:5], dif_power)
        return j * active, pos * active

    j, pos = fn(data, id2_t, dofadr1_t, dofadr2_t, qposadr1_t, qposadr2_t, active)
    invweight = m.dof_invweight0[dofadr1_t] + m.dof_invweight0[dofadr2_t] * (id2_t > -1)
    solref, solimp = m.eq_solref[ids], m.eq_solimp[ids]
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_limit_ball(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for ball joint limits."""
    _dev = d.qpos.device
    ids = precomp["ids"].to(_dev)
    qposadr = precomp["qposadr"].to(_dev)
    dofadr = precomp["dofadr"].to(_dev)
    dofadr_first = precomp["dofadr_first"].to(_dev)

    jnt_range = m.jnt_range[ids]
    jnt_margin = m.jnt_margin[ids]

    @torch.vmap
    def fn(jnt_range, jnt_margin, qposadr, dofadr):
        axis, angle = math.quat_to_axis_angle(torch.gather(d.qpos, 0, qposadr))
        j = torch.zeros(m.nv, dtype=jnt_range.dtype, device=jnt_range.device)
        j = j.scatter(0, dofadr, -axis)
        pos = torch.amax(jnt_range) - angle - jnt_margin
        active = pos < 0
        return j * active, pos * active

    j, pos = fn(jnt_range, jnt_margin, qposadr, dofadr)
    invweight = m.dof_invweight0[dofadr_first]
    solref, solimp = m.jnt_solref[ids], m.jnt_solimp[ids]
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_limit_slide_hinge(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for slide and hinge joint limits."""
    _dev = d.qpos.device
    ids = precomp["ids"].to(_dev)
    qposadr = precomp["qposadr"].to(_dev)
    dofadr = precomp["dofadr"].to(_dev)

    jnt_range = m.jnt_range[ids]
    jnt_margin = m.jnt_margin[ids]

    @torch.vmap
    def fn(jnt_range, jnt_margin, qposadr, dofadr):
        dist_min = d.qpos.gather(0, qposadr.unsqueeze(0)).squeeze(0) - jnt_range[0]
        dist_max = jnt_range[1] - d.qpos.gather(0, qposadr.unsqueeze(0)).squeeze(0)
        j = torch.zeros(m.nv, dtype=jnt_range.dtype, device=jnt_range.device)
        val = ((dist_min < dist_max).to(jnt_range.dtype) * 2 - 1).unsqueeze(0)
        j = j.scatter(0, dofadr.unsqueeze(0), val)
        pos = torch.minimum(dist_min, dist_max) - jnt_margin
        active = pos < 0
        return j * active, pos * active

    j, pos = fn(jnt_range, jnt_margin, qposadr, dofadr)
    invweight = m.dof_invweight0[dofadr]
    solref, solimp = m.jnt_solref[ids], m.jnt_solimp[ids]
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_limit_tendon(m: Model, d: Data, precomp: dict) -> _Efc:
    """Calculates constraint rows for tendon limits."""
    _dev = d.qpos.device
    tendon_id = precomp["tendon_id"].to(_dev)

    length = d.ten_length[tendon_id]
    j = d.ten_J[tendon_id]
    range_ = m.tendon_range[tendon_id]
    margin = m.tendon_margin[tendon_id]
    invweight = m.tendon_invweight0[tendon_id]
    solref = m.tendon_solref_lim[tendon_id]
    solimp = m.tendon_solimp_lim[tendon_id]

    dist_min = length - range_[:, 0]
    dist_max = range_[:, 1] - length
    pos = torch.minimum(dist_min, dist_max) - margin
    active = (pos < 0).to(j.dtype)
    sign = ((dist_min < dist_max).to(j.dtype) * 2 - 1) * active
    j = j * sign.unsqueeze(1)
    pos = pos * active
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_contact_frictionless(m: Model, d: Data) -> _Efc:
    """Calculates constraint rows for frictionless (condim=1) contacts.

    Each condim=1 contact produces a single normal-force constraint row.
    """
    ncon_fl, _ = m.condim_counts_py

    @torch.vmap
    def fn(c: Contact):
        dist = c.dist - c.includemargin
        g1 = c.geom1.long().unsqueeze(0)
        g2 = c.geom2.long().unsqueeze(0)
        body1 = m.geom_bodyid_t.gather(0, g1).squeeze(0)
        body2 = m.geom_bodyid_t.gather(0, g2).squeeze(0)
        diff = support.jac_dif_pair(m, d, c.pos, body1, body2)
        invweight0 = m.body_invweight0[:, 0]
        t = invweight0.gather(0, body1.unsqueeze(0)).squeeze(0) + invweight0.gather(0, body2.unsqueeze(0)).squeeze(0)

        j = (c.frame @ diff.T)[0]

        active = dist < 0
        return j * active, dist * active, t

    contact = d.contact
    if contact.batch_size != torch.Size([contact.dist.shape[0]]):
        contact = contact.clone(recurse=False)
        contact.auto_batch_size_()
    contact = contact[:ncon_fl]
    j, pos, invweight = fn(contact)
    solref = contact.solref
    solimp = contact.solimp
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


def _instantiate_contact(m: Model, d: Data) -> _Efc:
    """Calculates constraint rows for frictional (condim=3) pyramidal contacts."""
    ncon_fl, ncon_fr = m.condim_counts_py

    @torch.vmap
    def fn(c: Contact):
        dist = c.dist - c.includemargin
        g1 = c.geom1.long().unsqueeze(0)
        g2 = c.geom2.long().unsqueeze(0)
        body1 = m.geom_bodyid_t.gather(0, g1).squeeze(0)
        body2 = m.geom_bodyid_t.gather(0, g2).squeeze(0)
        diff = support.jac_dif_pair(m, d, c.pos, body1, body2)
        invweight0 = m.body_invweight0[:, 0]
        t = invweight0.gather(0, body1.unsqueeze(0)).squeeze(0) + invweight0.gather(0, body2.unsqueeze(0)).squeeze(0)

        # rotate Jacobian differences to contact frame
        diff_con = c.frame @ diff.T

        # 4 pyramidal friction directions
        js, invweights = [], []
        for diff_tan, friction in zip(diff_con[1:], c.friction[:2]):
            for f in (friction, -friction):
                js.append(diff_con[0] + diff_tan * f)
                invweights.append((t + f * f * t) * 2 * f * f)

        active = dist < 0
        j, invweight = torch.stack(js) * active, torch.stack(invweights)
        pos = dist.unsqueeze(0).expand(4) * active
        solref, solimp = torch.tile(c.solref, (4, 1)), torch.tile(c.solimp, (4, 1))

        return j, invweight, pos, solref, solimp

    contact = d.contact
    if contact.batch_size != torch.Size([contact.dist.shape[0]]):
        contact = contact.clone(recurse=False)
        contact.auto_batch_size_()
    contact = contact[ncon_fl : ncon_fl + ncon_fr]
    res = fn(contact)
    # remove contact grouping dimension: flatten (ncon, 4, ...) to (ncon*4, ...)
    j, invweight, pos, solref, solimp = torch.utils._pytree.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), res)
    frictionloss = torch.zeros_like(pos)

    return _Efc(
        J=j,
        pos=pos,
        pos_norm=pos,
        invweight=invweight,
        solref=solref,
        solimp=solimp,
        frictionloss=frictionloss,
        batch_size=[j.shape[0]],
    )


constraint_sizes = collision_driver.constraint_sizes


def count_constraints(m: Model, d: Data) -> tuple[int, int, int, int]:
    """Returns equality, friction, limit, and contact constraint counts.

    .. deprecated:: Use :func:`constraint_sizes` instead which does not
       require a ``Data`` instance and is compatible with ``torch.vmap``.
    """
    ne, nf, nl, ncon_, nefc = collision_driver.constraint_sizes(m)
    nc = nefc - ne - nf - nl
    return ne, nf, nl, nc


def make_constraint(m: Model, d: Data) -> Data:
    """Creates constraint jacobians and other supporting data."""
    ne, nf, nl, ncon, nefc = collision_driver.constraint_sizes(m)
    ns = ne + nf + nl
    actual_ncon = d.contact.dist.shape[0]
    has_contacts = ncon > 0 and actual_ncon > 0
    if has_contacts:
        dims = d.contact.contact_dim
        rows_per = torch.where(dims == 1, 1, (dims - 1) * 2)
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=rows_per.dtype, device=rows_per.device), rows_per[:-1]]),
            dim=0,
        )
        efc_address = (ns + offsets).to(torch.int64)
    else:
        efc_address = torch.empty(0, dtype=torch.int64)
    d = d.tree_replace({"contact.efc_address": efc_address})

    precomp = m.constraint_data_py
    ncon_fl, ncon_fr = m.condim_counts_py

    efcs = []
    if precomp["eq_connect"] is not None:
        efcs.append(_instantiate_equality_connect(m, d, precomp["eq_connect"]))
    if precomp["eq_weld"] is not None:
        efcs.append(_instantiate_equality_weld(m, d, precomp["eq_weld"]))
    if precomp["eq_joint"] is not None:
        efcs.append(_instantiate_equality_joint(m, d, precomp["eq_joint"]))
    if precomp["friction"] is not None:
        efcs.append(_instantiate_friction(m, d, precomp["friction"]))
    if precomp["limit_ball"] is not None:
        efcs.append(_instantiate_limit_ball(m, d, precomp["limit_ball"]))
    if precomp["limit_slide_hinge"] is not None:
        efcs.append(_instantiate_limit_slide_hinge(m, d, precomp["limit_slide_hinge"]))
    if precomp["limit_tendon"] is not None:
        efcs.append(_instantiate_limit_tendon(m, d, precomp["limit_tendon"]))
    if ncon_fl > 0 and has_contacts:
        efcs.append(_instantiate_contact_frictionless(m, d))
    if ncon_fr > 0 and has_contacts:
        efcs.append(_instantiate_contact(m, d))
    efcs = tuple(efcs)

    if not efcs:
        _dev = d.qpos.device
        z = torch.empty(0, device=_dev)
        d = d.replace(efc_J=torch.empty((0, m.nv), device=_dev))
        d = d.replace(efc_D=z, efc_aref=z, efc_frictionloss=z, nefc=_ZERO_I32.get(torch.int32, _dev))
        return d

    efc = torch.cat(list(efcs))
    refsafe = precomp["refsafe"]

    @torch.vmap
    def fn(efc):
        k, b, imp = _kbi(m, efc.solref, efc.solimp, efc.pos_norm, refsafe=refsafe)
        r = torch.maximum(
            efc.invweight * (1 - imp) / imp,
            _MJMINVAL.get(efc.invweight.dtype, efc.invweight.device),
        )
        aref = -b * (efc.J @ d.qvel) - k * imp * efc.pos
        return aref, r

    aref, r = fn(efc)
    d = d.replace(efc_J=efc.J, efc_D=1 / r, efc_aref=aref)
    d = d.replace(
        efc_frictionloss=efc.frictionloss,
        nefc=torch.full((), r.shape[0], dtype=torch.int32, device=r.device),
    )

    return d
