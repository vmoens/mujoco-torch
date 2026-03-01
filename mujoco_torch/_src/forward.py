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
"""Forward step functions."""

from collections.abc import Sequence

import mujoco

# pylint: enable=g-importing-member
import torch
from mujoco_torch._src import (
    collision_driver,
    constraint,
    derivative,
    math,
    passive,
    scan,
    sensor,
    smooth,
    solver,
    support,
)

# pylint: disable=g-importing-member
from mujoco_torch._src.types import BiasType, Data, DisableBit, DynType, GainType, IntegratorType, JointType, Model

# RK4 tableau (cached per device to avoid CPU→CUDA copies during graph capture)
_RK4_A = math._CachedConst(
    [
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
_RK4_B = math._CachedConst([1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0])


def _position(m: Model, d: Data) -> Data:
    """Position-dependent computations."""
    d = smooth.kinematics(m, d)
    d = smooth.com_pos(m, d)
    d = smooth.tendon(m, d)
    d = smooth.crb(m, d)
    d = smooth.tendon_armature(m, d)
    d = smooth.factor_m(m, d)
    d = collision_driver.collision(m, d)
    d = constraint.make_constraint(m, d)
    d = smooth.transmission(m, d)
    return d


def _velocity(m: Model, d: Data) -> Data:
    """Velocity-dependent computations."""
    actuator_moment = d.actuator_moment
    if actuator_moment.ndim == 1 and m.nu == 0:
        actuator_moment = actuator_moment.reshape(0, m.nv)
    kwargs = {"actuator_velocity": actuator_moment @ d.qvel}
    if m.ntendon:
        kwargs["ten_velocity"] = d.ten_J @ d.qvel
    d.update_(**kwargs)
    d = smooth.com_vel(m, d)
    d = passive.passive(m, d)
    d = smooth.rne(m, d)
    return d


def _actuation(m: Model, d: Data) -> Data:
    """Actuation-dependent computations."""
    if not m.nu or m.opt.disableflags & DisableBit.ACTUATION:
        d.update_(
            act_dot=torch.zeros((m.na,)),
            qfrc_actuator=torch.zeros((m.nv,)),
        )
        return d

    ctrl = d.ctrl
    if not m.opt.disableflags & DisableBit.CLAMPCTRL:
        ctrlrange = torch.where(
            m.actuator_ctrllimited_bool,
            m.actuator_ctrlrange,
            math._INF_RANGE.get(ctrl.dtype, ctrl.device),
        )
        # ctrl = torch.clamp(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])
        ctrl = torch.minimum(torch.maximum(ctrl, ctrlrange[:, 0]), ctrlrange[:, 1])

    # act_dot for stateful actuators
    def get_act_dot(dyn_typ, dyn_prm, ctrl, act):
        if dyn_typ == DynType.NONE:
            act_dot = torch.zeros_like(ctrl)
        elif dyn_typ == DynType.INTEGRATOR:
            act_dot = ctrl
        elif dyn_typ == DynType.FILTER:
            act_dot = (ctrl - act) / torch.clamp_min(dyn_prm[0], mujoco.mjMINVAL)
        else:
            raise NotImplementedError(f"dyntype {dyn_typ.name} not implemented.")
        return act_dot

    act_dot = torch.zeros((m.na,), dtype=d.qpos.dtype, device=d.qpos.device)
    if m.na:
        act_dot = scan.flat(
            m,
            get_act_dot,
            "uuua",
            "a",
            m.actuator_dyntype,
            m.actuator_dynprm,
            ctrl,
            d.act,
            group_by="u",
        )

    ctrl_act = ctrl
    if m.na:
        act_last_dim = d.act[m.actuator_actadr + m.actuator_actnum - 1]
        ctrl_act = torch.where(m.actuator_actadr_neg1, ctrl, act_last_dim)

    def get_force(*args):
        gain_t, gain_p, bias_t, bias_p, len_, vel, ctrl_act = args

        typ, prm = GainType(gain_t), gain_p
        if typ == GainType.FIXED:
            gain = prm[0]
        elif typ == GainType.AFFINE:
            gain = prm[0] + prm[1] * len_ + prm[2] * vel
        else:
            raise RuntimeError(f"unrecognized gaintype {typ.name}.")

        typ, prm = BiasType(bias_t), bias_p
        bias = torch.zeros_like(len_)
        if typ == BiasType.AFFINE:
            bias = prm[0] + prm[1] * len_ + prm[2] * vel

        return gain * ctrl_act + bias

    force = scan.flat(
        m,
        get_force,
        "uuuuuuu",
        "u",
        m.actuator_gaintype,
        m.actuator_gainprm,
        m.actuator_biastype,
        m.actuator_biasprm,
        d.actuator_length,
        d.actuator_velocity,
        ctrl_act,
        group_by="u",
    )
    forcerange = torch.where(
        m.actuator_forcelimited_bool,
        m.actuator_forcerange,
        math._INF_RANGE.get(force.dtype, force.device),
    )
    force = torch.clamp(force, forcerange[:, 0], forcerange[:, 1])

    qfrc_actuator = d.actuator_moment.T @ force

    # actuator-level gravity compensation
    if m.has_gravcomp:
        qfrc_actuator = qfrc_actuator + d.qfrc_gravcomp * m.jnt_actgravcomp[m.dof_jntid]

    # clamp qfrc_actuator
    actfrcrange = torch.where(
        m.jnt_actfrclimited_bool,
        m.jnt_actfrcrange,
        math._INF_RANGE.get(qfrc_actuator.dtype, qfrc_actuator.device),
    )
    actfrcrange = actfrcrange[m.dof_jntid_t]
    qfrc_actuator = torch.clamp(qfrc_actuator, actfrcrange[:, 0], actfrcrange[:, 1])

    d.update_(act_dot=act_dot, qfrc_actuator=qfrc_actuator)
    return d


def _acceleration(m: Model, d: Data) -> Data:
    """Add up all non-constraint forces, compute qacc_smooth."""
    qfrc_applied = d.qfrc_applied + support.xfrc_accumulate(m, d)
    qfrc_smooth = d.qfrc_passive - d.qfrc_bias + d.qfrc_actuator + qfrc_applied
    qacc_smooth = smooth.solve_m(m, d, qfrc_smooth)
    d.update_(qfrc_smooth=qfrc_smooth, qacc_smooth=qacc_smooth)
    return d


def _integrate_pos(jnt_typs: Sequence[str], qpos: torch.Tensor, qvel: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Integrate position given velocity."""
    qs, qi, vi = [], 0, 0

    for jnt_typ in jnt_typs:
        if jnt_typ == JointType.FREE:
            pos = qpos[qi : qi + 3] + dt * qvel[vi : vi + 3]
            quat = math.quat_integrate(qpos[qi + 3 : qi + 7], qvel[vi + 3 : vi + 6], dt)
            qs.append(torch.cat([pos, quat]))
            qi, vi = qi + 7, vi + 6
        elif jnt_typ == JointType.BALL:
            quat = math.quat_integrate(qpos[qi : qi + 4], qvel[vi : vi + 3], dt)
            qs.append(quat)
            qi, vi = qi + 4, vi + 3
        elif jnt_typ in (JointType.HINGE, JointType.SLIDE):
            pos = qpos[qi] + dt * qvel[vi]
            qs.append(pos[None])
            qi, vi = qi + 1, vi + 1
        else:
            raise RuntimeError(f"unrecognized joint type: {jnt_typ}")

    return torch.cat(qs) if qs else torch.empty((0,))


def _advance(
    m: Model,
    d: Data,
    act_dot: torch.Tensor,
    qacc: torch.Tensor,
    qvel: torch.Tensor | None = None,
) -> Data:
    """Advance state and time given activation derivatives and acceleration."""
    act = d.act
    if m.na:
        act = d.act + act_dot * m.opt.timestep
        actrange = torch.where(
            m.actuator_actlimited_bool,
            m.actuator_actrange,
            math._INF_RANGE.get(act.dtype, act.device),
        )
        fn = lambda act, actrange: torch.clamp(act, actrange[0], actrange[1])
        act = scan.flat(m, fn, "au", "a", act, actrange, group_by="u")

    # advance velocities
    new_qvel = d.qvel + qacc * m.opt.timestep

    # advance positions with qvel if given, new_qvel otherwise (semi-implicit)
    qvel_for_pos = new_qvel if qvel is None else qvel
    integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)
    qpos = scan.flat(m, integrate_fn, "jqv", "q", m.jnt_type, d.qpos, qvel_for_pos)

    # advance time
    time = d.time + m.opt.timestep

    d.update_(qvel=new_qvel, act=act, qpos=qpos, time=time)
    return d


def _euler(m: Model, d: Data) -> Data:
    """Euler integrator, semi-implicit in velocity."""
    # integrate damping implicitly
    qacc = d.qacc
    if not m.opt.disableflags & DisableBit.EULERDAMP:
        if support.is_sparse(m):
            qM = d.qM.clone()
            madr = m.dof_Madr_t
            qM = qM.index_add(0, madr, m.opt.timestep * m.dof_damping)
        else:
            qM = d.qM + torch.diag(m.opt.timestep * m.dof_damping)
        dh = d.replace(qM=qM)
        dh = smooth.factor_m(m, dh)
        qfrc = d.qfrc_smooth + d.qfrc_constraint
        qacc = smooth.solve_m(m, dh, qfrc)
    return _advance(m, d, d.act_dot, qacc)


def _rungekutta4(m: Model, d: Data, fixed_iterations: bool = False) -> Data:
    """Runge-Kutta explicit order 4 integrator."""
    d_t0 = d.clone(recurse=False)
    # pylint: disable=invalid-name
    A_t = _RK4_A.get(d.qpos.dtype, d.qpos.device)
    B_t = _RK4_B.get(d.qpos.dtype, d.qpos.device)
    C = torch.tril(A_t).sum(dim=0)  # C(i) = sum_j A(i,j)
    T = d.time + C * m.opt.timestep
    # pylint: enable=invalid-name

    kqvel = d.qvel  # intermediate RK solution
    # RK solutions sum
    qvel, qacc, act_dot = torch.utils._pytree.tree_map(lambda k: B_t[0] * k, (kqvel, d.qacc, d.act_dot))
    integrate_fn = lambda *args: _integrate_pos(*args, dt=m.opt.timestep)

    # Unrolled loop (3 iterations) instead of torch_scan HOP.
    # The body calls forward() which accesses module-level scan caches — these
    # are "unsafe side effects" that break torch.ops.higher_order.scan under
    # fullgraph compilation.  Unrolling lets Dynamo trace each iteration at the
    # top level where dict access is fine.
    a_diag = torch.diag(A_t)
    for i in range(3):
        a, b, t = a_diag[i], B_t[i + 1], T[i]
        dqvel, dqacc, dact_dot = torch.utils._pytree.tree_map(lambda k: a * k, (kqvel, d.qacc, d.act_dot))
        kqpos = scan.flat(m, integrate_fn, "jqv", "q", m.jnt_type, d_t0.qpos, dqvel)
        kact = d_t0.act + dact_dot * m.opt.timestep
        kqvel = d_t0.qvel + dqacc * m.opt.timestep
        d.update_(qpos=kqpos, qvel=kqvel, act=kact, time=t)
        d = forward(m, d, fixed_iterations=fixed_iterations)

        qvel = qvel + b * kqvel
        qacc = qacc + b * d.qacc
        act_dot = act_dot + b * d.act_dot

    d = _advance(m, d_t0, act_dot, qacc, qvel)
    return d


def forward(m: Model, d: Data, fixed_iterations: bool = False) -> Data:
    """Forward dynamics.

    Args:
      m: Model.
      d: Data.
      fixed_iterations: when True, the constraint solver runs a fixed number
        of iterations (no early termination), producing a static computation
        graph suitable for CUDA graph capture.
    """
    d = _position(m, d)
    d = sensor.sensor_pos(m, d)
    d = _velocity(m, d)
    d = sensor.sensor_vel(m, d)
    d = _actuation(m, d)
    d = _acceleration(m, d)

    if d.efc_J.numel() == 0:
        d.update_(qacc=d.qacc_smooth)
        return d

    d = solver.solve(m, d, fixed_iterations=fixed_iterations)
    d = sensor.sensor_acc(m, d)

    return d


def _implicit(m: Model, d: Data) -> Data:
    """Integrates fully implicit in velocity."""
    qderiv = derivative.deriv_smooth_vel(m, d)

    qacc = d.qacc
    if qderiv is not None:
        qm = support.full_m(m, d) if support.is_sparse(m) else d.qM.clone()
        qm = qm - m.opt.timestep * qderiv
        L = torch.linalg.cholesky(qm)
        qfrc = d.qfrc_smooth + d.qfrc_constraint
        qacc = torch.cholesky_solve(qfrc.unsqueeze(-1), L).squeeze(-1)

    return _advance(m, d, d.act_dot, qacc)


def step(m: Model, d: Data, fixed_iterations: bool = False) -> Data:
    """Advance simulation.

    Args:
      m: Model.
      d: Data.
      fixed_iterations: when True, the constraint solver runs a fixed number
        of iterations (no early termination), producing a static computation
        graph suitable for CUDA graph capture.
    """
    d = d.clone(recurse=False)
    d = forward(m, d, fixed_iterations=fixed_iterations)

    if m.opt.integrator == IntegratorType.EULER:
        d = _euler(m, d)
    elif m.opt.integrator == IntegratorType.RK4:
        d = _rungekutta4(m, d, fixed_iterations=fixed_iterations)
    elif m.opt.integrator == IntegratorType.IMPLICITFAST:
        d = _implicit(m, d)
    else:
        raise NotImplementedError(f"integrator {m.opt.integrator} not implemented.")

    return d


# Public aliases
fwd_position = _position
fwd_velocity = _velocity
fwd_actuation = _actuation
fwd_acceleration = _acceleration
euler = _euler
rungekutta4 = _rungekutta4
implicit = _implicit
