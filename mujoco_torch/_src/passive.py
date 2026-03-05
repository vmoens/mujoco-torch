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
"""Passive forces."""

import torch

from mujoco_torch._src.math import _CachedConst

_EPS12 = _CachedConst(1e-12)

from mujoco_torch._src import math, scan, support

# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data, DisableBit, JointType, Model

_FREE = JointType.FREE
_BALL = JointType.BALL

# pylint: enable=g-importing-member


def _inertia_box_fluid_model(
    m: Model,
    inertia: torch.Tensor,
    mass: torch.Tensor,
    root_com: torch.Tensor,
    xipos: torch.Tensor,
    ximat: torch.Tensor,
    cvel: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fluid forces based on inertia-box approximation."""
    box = inertia.unsqueeze(0).repeat(3, 1)
    box = box * (torch.ones((3, 3), dtype=inertia.dtype, device=inertia.device) - 2 * torch.eye(3))
    box = 6.0 * torch.clamp(torch.sum(box, dim=-1), min=1e-12)
    box = torch.sqrt(box / torch.maximum(mass, _EPS12.get(mass.dtype, mass.device))) * (mass > 0.0)

    # transform to local coordinate frame
    offset = xipos - root_com
    lvel = math.transform_motion(cvel, offset, ximat)
    lwind = ximat.T @ m.opt.wind
    lvel = lvel.clone()
    lvel[3:] = lvel[3:] + (-lwind)

    # set viscous force and torque
    diam = torch.mean(box, dim=-1)
    lfrc_ang = lvel[:3] * -torch.pi * diam**3 * m.opt.viscosity
    lfrc_vel = lvel[3:] * -3.0 * torch.pi * diam * m.opt.viscosity

    # add lift and drag force and torque
    scale_vel = torch.stack(
        [box[1] * box[2], box[0] * box[2], box[0] * box[1]],
    )
    scale_ang = torch.stack(
        [
            box[0] * (box[1] ** 4 + box[2] ** 4),
            box[1] * (box[0] ** 4 + box[2] ** 4),
            box[2] * (box[0] ** 4 + box[1] ** 4),
        ]
    )
    lfrc_vel = lfrc_vel - 0.5 * m.opt.density * scale_vel * torch.abs(lvel[3:]) * lvel[3:]
    lfrc_ang = lfrc_ang - (1.0 * m.opt.density * scale_ang * torch.abs(lvel[:3]) * lvel[:3] / 64.0)

    # rotate to global orientation: lfrc -> bfrc
    force, torque = ximat @ lfrc_vel, ximat @ lfrc_ang

    return force, torque


def _spring_damper(m: Model, d: Data) -> torch.Tensor:
    """Applies joint level spring and damping forces."""

    max_j = m.max_joints_per_body
    has_free = _FREE in m.joint_types_present
    has_ball = _BALL in m.joint_types_present
    max_dw = m.max_dof_per_jnt

    def fn(jnt_typs, stiffness, qpos_spring, qpos):
        dtype, device = qpos.dtype, qpos.device
        pad = max_dw - 1
        qfrc_list = []

        for j in range(max_j):
            jt = jnt_typs[j]
            q = qpos[j]
            qs = qpos_spring[j]
            s = stiffness[j]

            # HINGE/SLIDE: 1 DOF + pad zeros
            hs_val = -s * (q[0] - qs[0])
            qfrc = torch.cat([hs_val.unsqueeze(0), torch.zeros(pad, dtype=dtype, device=device)])

            if has_ball:
                ball_rot = -s * math.quat_sub(q[:4], qs[:4])
                ball_qfrc = torch.cat([ball_rot, torch.zeros(max_dw - 3, dtype=dtype, device=device)])
                qfrc = torch.where(jt == _BALL, ball_qfrc, qfrc)

            if has_free:
                free_trans = -s * (q[:3] - qs[:3])
                free_rot = -s * math.quat_sub(q[3:7], qs[3:7])
                free_qfrc = torch.cat([free_trans, free_rot])
                qfrc = torch.where(jt == _FREE, free_qfrc, qfrc)

            qfrc_list.append(qfrc)

        return torch.stack(qfrc_list)

    qfrc = torch.zeros(m.nv, dtype=d.qpos.dtype, device=d.qpos.device)

    if not m.opt.disableflags & DisableBit.SPRING:
        qfrc = qfrc + scan.flat(
            m,
            fn,
            "jjqq",
            "v",
            m.jnt_type,
            m.jnt_stiffness,
            m.qpos_spring,
            d.qpos,
        )

    if not m.opt.disableflags & DisableBit.DAMPER:
        qfrc = qfrc - m.dof_damping * d.qvel

    # tendon-level springs
    if m.ntendon and not m.opt.disableflags & DisableBit.SPRING:
        below = m.tendon_lengthspring[:, 0] - d.ten_length
        above = m.tendon_lengthspring[:, 1] - d.ten_length
        frc_spring = torch.where(below > 0, m.tendon_stiffness * below, torch.zeros_like(below))
        frc_spring = torch.where(above < 0, m.tendon_stiffness * above, frc_spring)
    else:
        frc_spring = torch.zeros(max(m.ntendon, 0), dtype=d.qpos.dtype, device=d.qpos.device)

    # tendon-level dampers
    if m.ntendon and not m.opt.disableflags & DisableBit.DAMPER:
        frc_damper = -m.tendon_damping * d.ten_velocity
    else:
        frc_damper = torch.zeros(max(m.ntendon, 0), dtype=d.qpos.dtype, device=d.qpos.device)

    if m.ntendon:
        qfrc = qfrc + d.ten_J.T @ (frc_spring + frc_damper)

    return qfrc


def _gravcomp(m: Model, d: Data) -> torch.Tensor:
    """Applies body-level gravity compensation."""
    force = -m.opt.gravity * (m.body_mass * m.body_gravcomp).unsqueeze(-1)

    apply_f = lambda f, pos, body_id: support.jac(m, d, pos, body_id)[0] @ f
    qfrc = torch.vmap(apply_f)(force, d.xipos, torch.arange(m.nbody, device=d.xipos.device)).sum(dim=0)

    return qfrc


def _fluid(m: Model, d: Data) -> torch.Tensor:
    """Applies body-level viscosity, lift and drag."""
    force, torque = torch.vmap(_inertia_box_fluid_model, (None, 0, 0, 0, 0, 0, 0))(
        m,
        m.body_inertia,
        m.body_mass,
        d.subtree_com[m.body_rootid_t],
        d.xipos,
        d.ximat,
        d.cvel,
    )
    qfrc = torch.vmap(support.apply_ft, (None, None, 0, 0, 0, 0))(
        m, d, force, torque, d.xipos, torch.arange(m.nbody, device=d.xipos.device)
    )

    return torch.sum(qfrc, dim=0)


def passive(m: Model, d: Data) -> Data:
    """Adds all passive forces."""
    if m.opt.disableflags & (DisableBit.SPRING | DisableBit.DAMPER):
        d.update_(
            qfrc_passive=torch.zeros(m.nv, dtype=d.qpos.dtype, device=d.qpos.device),
            qfrc_gravcomp=torch.zeros(m.nv, dtype=d.qpos.dtype, device=d.qpos.device),
        )
        return d

    qfrc_passive = _spring_damper(m, d)
    qfrc_gravcomp = torch.zeros(m.nv, dtype=d.qpos.dtype, device=d.qpos.device)

    if m.has_gravcomp and not m.opt.disableflags & DisableBit.GRAVITY:
        qfrc_gravcomp = _gravcomp(m, d)
        qfrc_passive = qfrc_passive + qfrc_gravcomp * (1 - m.jnt_actgravcomp[m.dof_jntid])

    if m.opt.has_fluid_params:
        qfrc_passive = qfrc_passive + _fluid(m, d)

    d.update_(qfrc_passive=qfrc_passive, qfrc_gravcomp=qfrc_gravcomp)
    return d
