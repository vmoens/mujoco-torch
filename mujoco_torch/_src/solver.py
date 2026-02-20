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
"""Constraint solvers."""

from typing import NamedTuple

import mujoco
import numpy as np
import torch
from torch._higher_order_ops.while_loop import while_loop as _torch_while_loop


def _inside_functorch() -> bool:
    """Return True when executing inside a functorch transform (vmap, grad, …)."""
    return torch._C._functorch.peek_interpreter_stack() is not None


def while_loop(cond_fn, body_fn, carried_inputs, max_iter=None):
    """While loop that dispatches appropriately depending on context.

    * ``torch.compile`` or ``torch.vmap`` → ``torch._higher_order_ops.while_loop``
      (the higher-order op has both compile and vmap dispatch rules as of
      torch 2.11; the vmap rule runs until all batch elements converge,
      freezing finished elements with ``torch.where``).
    * eager → plain Python ``while``

    Args:
      cond_fn:  ``(*carried) -> bool`` – loop condition.
      body_fn:  ``(*carried) -> carried`` – loop body.
      carried_inputs: initial carry (tuple of tensors / NamedTuples).
      max_iter: unused, kept for backward compatibility.
    """
    if torch.compiler.is_compiling() or _inside_functorch():
        # Wrap body to clone all output tensors, eliminating input-to-output
        # aliasing (required by torch._higher_order_ops.while_loop when carry
        # fields pass through unchanged).
        orig_body_fn = body_fn

        def cloning_body(*args):
            result = orig_body_fn(*args)
            return torch.utils._pytree.tree_map(
                lambda t: t.clone() if isinstance(t, torch.Tensor) else t,
                result,
            )

        return _torch_while_loop(cond_fn, cloning_body, carried_inputs)

    val = carried_inputs
    while cond_fn(*val):
        val = body_fn(*val)
        if not isinstance(val, tuple):
            val = (val,)
    return val


from mujoco_torch._src import math, support
from mujoco_torch._src.collision_driver import constraint_sizes

# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data, DisableBit, Model, SolverType

# pylint: enable=g-importing-member


# ============================================================================
# NamedTuples for solver state.
# NamedTuples are natively understood by torch.compile / while_loop as pytrees,
# unlike custom dataclass-registered pytrees which cause tracing errors.
# ============================================================================


class _Context(NamedTuple):
    """Data updated during each solver iteration.

    Attributes:
      qacc: acceleration (from Data)                    (nv,)
      qfrc_constraint: constraint force (from Data)     (nv,)
      Jaref: Jac*qacc - aref                            (nefc,)
      efc_force: constraint force in constraint space   (nefc,)
      Ma: M*qacc                                        (nv,)
      grad: gradient of master cost                     (nv,)
      Mgrad: M / grad                                   (nv,)
      search: linesearch vector                         (nv,)
      gauss: gauss Cost
      cost: constraint + Gauss cost
      prev_cost: cost from previous iter
      solver_niter: number of solver iterations
    """

    qacc: torch.Tensor
    qfrc_constraint: torch.Tensor
    Jaref: torch.Tensor  # pylint: disable=invalid-name
    efc_force: torch.Tensor
    Ma: torch.Tensor  # pylint: disable=invalid-name
    grad: torch.Tensor
    Mgrad: torch.Tensor  # pylint: disable=invalid-name
    search: torch.Tensor
    gauss: torch.Tensor
    cost: torch.Tensor
    prev_cost: torch.Tensor
    solver_niter: torch.Tensor


class _LSPoint(NamedTuple):
    """Line search evaluation point.

    Attributes:
      alpha: step size that reduces f(x + alpha * p) given search direction p
      cost: line search cost
      deriv_0: first derivative of quadratic
      deriv_1: second derivative of quadratic
    """

    alpha: torch.Tensor
    cost: torch.Tensor
    deriv_0: torch.Tensor
    deriv_1: torch.Tensor


class _LSContext(NamedTuple):
    """Data updated during each line search iteration.

    Attributes:
      lo: low point bounding the line search interval
      hi: high point bounding the line search interval
      swap: True if low or hi was swapped in the line search iteration
      ls_iter: number of linesearch iterations
    """

    lo: _LSPoint
    hi: _LSPoint
    swap: torch.Tensor
    ls_iter: torch.Tensor


# ============================================================================
# Mass matrix operation closures: precompute sparse indices at solve() entry
# so that the while_loop body never touches Model or Data TensorClasses.
# ============================================================================


def _make_solve_m_fn(m: Model, d: Data):
    """Return a closure that computes inv(L'*D*L)*x without accessing m/d."""
    if not support.is_sparse(m):
        qLD = d.qLD.clone()

        def solve_m_fn(x):
            return torch.cholesky_solve(x.unsqueeze(-1), qLD).squeeze(-1)

        return solve_m_fn

    # Sparse path: precompute update index arrays from model topology.
    qLD = d.qLD.clone()
    qLDiagInv = d.qLDiagInv.clone()

    depth = []
    for i in range(m.nv):
        depth.append(depth[m.dof_parentid[i]] + 1 if m.dof_parentid[i] != -1 else 0)
    updates_i, updates_j = {}, {}
    for i in range(m.nv):
        madr_ij, j = m.dof_Madr[i], i
        while True:
            madr_ij, j = madr_ij + 1, m.dof_parentid[j]
            if j == -1:
                break
            updates_i.setdefault(depth[i], []).append((i, madr_ij, j))
            updates_j.setdefault(depth[j], []).append((j, madr_ij, i))

    # Pre-convert to tensors so the loop body is pure torch ops.
    j_updates = []
    for _, vals in sorted(updates_j.items(), reverse=True):
        arr = np.array(vals)
        j_updates.append(
            (
                torch.tensor(arr[:, 0], dtype=torch.long),
                torch.tensor(arr[:, 1], dtype=torch.long),
                torch.tensor(arr[:, 2], dtype=torch.long),
            )
        )

    i_updates = []
    for _, vals in sorted(updates_i.items()):
        arr = np.array(vals)
        i_updates.append(
            (
                torch.tensor(arr[:, 0], dtype=torch.long),
                torch.tensor(arr[:, 1], dtype=torch.long),
                torch.tensor(arr[:, 2], dtype=torch.long),
            )
        )

    def solve_m_fn(x):
        # x <- inv(L') * x
        for j_idx, madr_idx, i_idx in j_updates:
            update = x[j_idx] + (-qLD[madr_idx] * x[i_idx])
            x = x.scatter(0, j_idx, update)
        # x <- inv(D) * x
        x = x * qLDiagInv
        # x <- inv(L) * x
        for i_idx, madr_idx, j_idx in i_updates:
            update = x[i_idx] + (-qLD[madr_idx] * x[j_idx])
            x = x.scatter(0, i_idx, update)
        return x

    return solve_m_fn


def _make_mul_m_fn(m: Model, d: Data):
    """Return a closure that computes M*vec without accessing m/d."""
    if not support.is_sparse(m):
        qM = d.qM.clone()

        def mul_m_fn(vec):
            return qM @ vec

        return mul_m_fn

    # Sparse path: precompute index arrays.
    qM = d.qM.clone()
    dof_Madr_t = torch.tensor(m.dof_Madr, dtype=torch.long)

    is_, js, madr_ijs = [], [], []
    for i in range(m.nv):
        madr_ij, j = m.dof_Madr[i], i
        while True:
            madr_ij, j = madr_ij + 1, m.dof_parentid[j]
            if j == -1:
                break
            is_.append(i)
            js.append(j)
            madr_ijs.append(madr_ij)

    i_t = torch.tensor(is_, dtype=torch.long)
    j_t = torch.tensor(js, dtype=torch.long)
    madr_t = torch.tensor(madr_ijs, dtype=torch.long)

    def mul_m_fn(vec):
        diag_mul = qM[dof_Madr_t] * vec
        out = diag_mul.clone()
        out = out.index_add(0, i_t, qM[madr_t] * vec[j_t])
        out = out.index_add(0, j_t, qM[madr_t] * vec[i_t])
        return out

    return mul_m_fn


def _make_dense_m(m: Model, d: Data) -> torch.Tensor:
    """Pre-compute the dense mass matrix without later m/d access."""
    if not support.is_sparse(m):
        return d.qM.clone()

    # Sparse path: build dense from sparse representation.
    is_, js, madr_ijs = [], [], []
    for i in range(m.nv):
        madr_ij, j = m.dof_Madr[i], i
        while True:
            madr_ij, j = madr_ij + 1, m.dof_parentid[j]
            if j == -1:
                break
            is_.append(i)
            js.append(j)
            madr_ijs.append(madr_ij)

    i_idx = torch.tensor(is_, dtype=torch.int32, device=d.qM.device)
    j_idx = torch.tensor(js, dtype=torch.int32, device=d.qM.device)
    madr_idx = torch.tensor(madr_ijs, dtype=torch.int32, device=d.qM.device)

    mat = torch.zeros((m.nv, m.nv), dtype=d.qM.dtype, device=d.qM.device)
    mat[(i_idx, j_idx)] = d.qM[madr_idx]
    mat = torch.diag(d.qM[torch.tensor(m.dof_Madr)]) + mat + mat.T
    return mat


# ============================================================================
# Solver
# ============================================================================


def solve(m: Model, d: Data) -> Data:
    """Finds forces that satisfy constraints using conjugate gradient descent."""

    # ---- Pre-extract all model constants (become Dynamo compile-time consts) ---
    nv = int(m.nv)
    use_dense = nv < 100
    solver_type = m.opt.solver
    tolerance = float(m.opt.tolerance)
    iterations = int(m.opt.iterations)
    ls_tolerance = float(m.opt.ls_tolerance)
    ls_iterations = int(m.opt.ls_iterations)
    meaninertia = float(m.stat.meaninertia)
    disableflags = int(m.opt.disableflags)

    # ---- Pre-extract all Data tensors (become Dynamo graph inputs) ------------
    # Clone to guarantee no aliasing between closure variables and carry tensors
    # (required by torch._higher_order_ops.while_loop).
    efc_J = d.efc_J.clone()
    efc_J_T = efc_J.T.contiguous()  # separate tensor to avoid view aliasing
    efc_D = d.efc_D.clone()
    efc_aref = d.efc_aref.clone()
    qfrc_smooth = d.qfrc_smooth.clone()
    qacc_smooth = d.qacc_smooth.clone()
    qacc_warmstart = d.qacc_warmstart.clone()
    qfrc_constraint = d.qfrc_constraint.clone()
    ne, nf, nl, ncon_m, nefc = constraint_sizes(m)
    ne_nf = ne + nf

    # ---- Pre-compute mass matrix closures (no m/d access inside loop) ---------
    dense_M = _make_dense_m(m, d) if use_dense else torch.empty(0, 0)
    solve_m_fn = _make_solve_m_fn(m, d)
    mul_m_fn = _make_mul_m_fn(m, d) if not use_dense else None

    # ---- Inner helpers as closures over pre-extracted data --------------------

    def _rescale(value):
        return value / (meaninertia * max(1, nv))

    def _create_context(qacc, qfrc_con, grad_flag=True):
        """Create solver context from qacc without touching Model/Data."""
        jaref = efc_J @ qacc.to(efc_J.dtype) - efc_aref
        ma = dense_M @ qacc if use_dense else mul_m_fn(qacc)
        dtype = qacc.dtype
        ctx = _Context(
            qacc=qacc,
            qfrc_constraint=qfrc_con,
            Jaref=jaref,
            efc_force=torch.zeros(nefc, dtype=dtype),
            Ma=ma,
            grad=torch.zeros((nv,), dtype=dtype),
            Mgrad=torch.zeros((nv,), dtype=dtype),
            search=torch.zeros((nv,), dtype=dtype),
            gauss=torch.tensor(0.0, dtype=dtype),
            cost=torch.tensor(float("inf"), dtype=dtype),
            prev_cost=torch.tensor(0.0, dtype=dtype),
            solver_niter=torch.tensor(0),
        )
        ctx = _update_constraint(ctx)
        if grad_flag:
            ctx = _update_gradient(ctx)
            ctx = ctx._replace(search=-ctx.Mgrad)
        return ctx

    def _update_constraint(ctx):
        """Update constraint force and cost (no m/d access)."""
        active = ctx.Jaref < 0
        eq_fric_mask = torch.arange(active.shape[0], device=active.device) < ne_nf
        active = active | eq_fric_mask

        efc_force = efc_D * -ctx.Jaref * active
        qfrc_con = efc_J_T @ efc_force
        gauss = 0.5 * torch.dot(ctx.Ma - qfrc_smooth, ctx.qacc - qacc_smooth)
        cost = 0.5 * torch.sum(efc_D * ctx.Jaref * ctx.Jaref * active) + gauss

        return ctx._replace(
            qfrc_constraint=qfrc_con,
            gauss=gauss,
            cost=cost,
            prev_cost=ctx.cost,
            efc_force=efc_force,
        )

    def _update_gradient(ctx):
        """Update grad and M/grad (no m/d access)."""
        grad = ctx.Ma - qfrc_smooth - ctx.qfrc_constraint

        if solver_type == SolverType.CG:
            mgrad = solve_m_fn(grad)
        elif solver_type == SolverType.NEWTON:
            active = ctx.Jaref < 0
            eq_fric_mask = torch.arange(active.shape[0], device=active.device) < ne_nf
            active = active | eq_fric_mask
            h = (efc_J_T * efc_D * active) @ efc_J
            h = dense_M + h
            L = torch.linalg.cholesky(h)
            mgrad = torch.cholesky_solve(grad.unsqueeze(-1), L).squeeze(-1)
        else:
            raise NotImplementedError(f"unsupported solver type: {solver_type}")

        return ctx._replace(grad=grad, Mgrad=mgrad)

    def _linesearch(ctx):
        """Line search to find optimal step size (no m/d access)."""
        smag = math.norm(ctx.search) * meaninertia * max(1, nv)
        gtol = tolerance * ls_tolerance * smag

        mv = dense_M @ ctx.search if use_dense else mul_m_fn(ctx.search)
        jv = efc_J @ ctx.search

        quad_gauss = torch.stack(
            (
                ctx.gauss,
                torch.dot(ctx.search, ctx.Ma) - torch.dot(ctx.search, qfrc_smooth),
                0.5 * torch.dot(ctx.search, mv),
            )
        )
        quad = torch.stack((0.5 * ctx.Jaref * ctx.Jaref, jv * ctx.Jaref, 0.5 * jv * jv))
        quad = (quad * efc_D).T

        def point_fn(alpha):
            active = (ctx.Jaref + alpha * jv) < 0
            eq_fric_mask = torch.arange(active.shape[0], device=active.device) < ne_nf
            active = active | eq_fric_mask
            quad_active = quad * active.unsqueeze(1)
            quad_total = quad_gauss + torch.sum(quad_active, axis=0)

            cost = alpha * alpha * quad_total[2] + alpha * quad_total[1] + quad_total[0]
            deriv_0 = 2 * alpha * quad_total[2] + quad_total[1]
            deriv_1 = 2 * quad_total[2]
            return _LSPoint(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)

        def ls_cond(ls_ctx):
            done = ls_ctx.ls_iter >= ls_iterations
            done = done | (~ls_ctx.swap)
            done = done | ((ls_ctx.lo.deriv_0 < 0) & (ls_ctx.lo.deriv_0 > -gtol))
            done = done | ((ls_ctx.hi.deriv_0 > 0) & (ls_ctx.hi.deriv_0 < gtol))
            return ~done

        def ls_body(ls_ctx):
            lo, hi = ls_ctx.lo, ls_ctx.hi
            lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
            hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
            mid = point_fn(0.5 * (lo.alpha + hi.alpha))

            swap_lo_next = (lo.deriv_0 > 0) | (lo.deriv_0 < lo_next.deriv_0)
            lo = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_lo_next, y, x), lo, lo_next)
            swap_lo_mid = (mid.deriv_0 < 0) & (lo.deriv_0 < mid.deriv_0)
            lo = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_lo_mid, y, x), lo, mid)

            swap_hi_next = (hi.deriv_0 < 0) | (hi.deriv_0 > hi_next.deriv_0)
            hi = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_hi_next, y, x), hi, hi_next)
            swap_hi_mid = (mid.deriv_0 > 0) & (hi.deriv_0 > mid.deriv_0)
            hi = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_hi_mid, y, x), hi, mid)

            swap = swap_lo_next | swap_lo_mid | swap_hi_next | swap_hi_mid
            return (_LSContext(lo=lo, hi=hi, swap=swap, ls_iter=ls_ctx.ls_iter + 1),)

        # initialize interval
        p0 = point_fn(torch.tensor(0.0))
        lo = point_fn(p0.alpha - p0.deriv_0 / p0.deriv_1)
        lesser_fn = lambda x, y: torch.where(lo.deriv_0 < p0.deriv_0, x, y)
        hi = torch.utils._pytree.tree_map(lesser_fn, p0, lo)
        lo = torch.utils._pytree.tree_map(lesser_fn, lo, p0)
        ls_ctx = _LSContext(lo=lo, hi=hi, swap=torch.tensor(True), ls_iter=torch.tensor(0))
        ls_ctx = while_loop(ls_cond, ls_body, (ls_ctx,), max_iter=ls_iterations)[0]

        # move to new solution if improved
        lo, hi = ls_ctx.lo, ls_ctx.hi
        improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)
        alpha = torch.where(lo.cost < hi.cost, lo.alpha, hi.alpha)
        qacc = ctx.qacc + improved * ctx.search * alpha
        ma = ctx.Ma + improved * mv * alpha
        jaref = ctx.Jaref + improved * jv * alpha

        return ctx._replace(qacc=qacc, Ma=ma, Jaref=jaref)

    # ---- Main CG solver loop ----

    def cond(ctx):
        improvement = _rescale(ctx.prev_cost - ctx.cost)
        gradient = _rescale(torch.norm(ctx.grad))

        done = ctx.solver_niter >= iterations
        done = done | (improvement < tolerance)
        done = done | (gradient < tolerance)
        return ~done

    def body(ctx):
        ctx = _linesearch(ctx)
        prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad  # pylint: disable=invalid-name
        ctx = _update_constraint(ctx)
        ctx = _update_gradient(ctx)

        # polak-ribiere:
        beta = torch.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
        beta = beta / torch.clamp_min(torch.dot(prev_grad, prev_Mgrad), mujoco.mjMINVAL)
        beta = torch.clamp_min(beta, 0)
        search = -ctx.Mgrad + beta * ctx.search
        return (ctx._replace(search=search, solver_niter=ctx.solver_niter + 1),)

    # warmstart:
    qacc = qacc_smooth
    if not disableflags & DisableBit.WARMSTART:
        warm = _create_context(qacc_warmstart, qfrc_constraint, grad_flag=False)
        smth = _create_context(qacc_smooth, qfrc_constraint, grad_flag=False)
        qacc = torch.where(warm.cost < smth.cost, qacc_warmstart, qacc_smooth)

    ctx = _create_context(qacc, qfrc_constraint)
    if iterations == 1:
        ctx = body(ctx)[0]
    else:
        ctx = while_loop(cond, body, (ctx,), max_iter=iterations)[0]

    d = d.replace(
        qacc_warmstart=ctx.qacc,
        qacc=ctx.qacc,
        qfrc_constraint=ctx.qfrc_constraint,
        efc_force=ctx.efc_force,
    )

    return d
