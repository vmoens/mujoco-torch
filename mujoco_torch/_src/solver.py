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

from typing import Optional

# from torch import numpy as torch
import mujoco
import torch
from mujoco_torch._src import math
from mujoco_torch._src import smooth
# pylint: disable=g-importing-member
from mujoco_torch._src.dataclasses import PyTreeNode
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import Model
from mujoco_torch._src.types import SolverType


# pylint: enable=g-importing-member


class _Context(PyTreeNode):
  """Data updated during each solver iteration.

  Attributes:
    qacc: acceleration (from Data)                    (nv,)
    qfrc_constraint: constraint force (from Data)     (nv,)
    Jaref: Jac*qacc - aref                            (nefc,)
    efc_force: constraint force in constraint space   (nefc,)
    M: dense mass matrix, populated for nv < 100      (nv, nv)
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
  M: Optional[torch.Tensor]
  Ma: torch.Tensor  # pylint: disable=invalid-name
  grad: torch.Tensor
  Mgrad: torch.Tensor  # pylint: disable=invalid-name
  search: torch.Tensor
  gauss: torch.Tensor
  cost: torch.Tensor
  prev_cost: torch.Tensor
  solver_niter: torch.Tensor

  @classmethod
  def create(cls, m: Model, d: Data, grad: bool = True) -> '_Context':
    jaref = d.efc_J @ d.qacc - d.efc_aref
    # TODO(robotics-team): determine nv at which sparse mul is faster
    M = smooth.dense_m(m, d) if m.nv < 100 else None  # pylint: disable=invalid-name
    ma = smooth.mul_m(m, d, d.qacc) if M is None else M @ d.qacc
    nv_0 = torch.zeros((m.nv,))
    ctx = _Context(
        qacc=d.qacc,
        qfrc_constraint=d.qfrc_constraint,
        Jaref=jaref,
        efc_force=torch.zeros(d.nefc),
        M=M,
        Ma=ma,
        grad=nv_0,
        Mgrad=nv_0,
        search=nv_0,
        gauss=0.0,
        cost=torch.inf,
        prev_cost=0.0,
        solver_niter=0,
    )
    ctx = _update_constraint(m, d, ctx)
    if grad:
      ctx = _update_gradient(m, d, ctx)
      ctx = ctx.replace(search=-ctx.Mgrad)  # start with preconditioned gradient

    return ctx


class _LSPoint(PyTreeNode):
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

  @classmethod
  def create(
      cls,
      d: Data,
      ctx: _Context,
      alpha: torch.Tensor,
      jv: torch.Tensor,
      quad: torch.Tensor,
      quad_gauss: torch.Tensor,
  ) -> '_LSPoint':
    """Creates a linesearch point with first and second derivatives."""
    # roughly corresponds to CGEval in mujoco/src/engine/engine_solver.c

    # TODO(robotics-team): change this to support friction constraints
    active = ((ctx.Jaref + alpha * jv) < 0)
    active[:d.ne + d.nf] = True
    quad = torch.vmap(torch.multiply, (1, None), (1,))(quad, active)  # only active
    quad_total = quad_gauss + torch.sum(quad, axis=0)

    cost = alpha * alpha * quad_total[2] + alpha * quad_total[1] + quad_total[0]
    deriv_0 = 2 * alpha * quad_total[2] + quad_total[1]
    deriv_1 = 2 * quad_total[2]
    return _LSPoint(alpha=alpha, cost=cost, deriv_0=deriv_0, deriv_1=deriv_1)


class _LSContext(PyTreeNode):
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


def _while_loop_scan(cond_fun, body_fun, init_val, max_iter):
  """Scan-based implementation (jit ok, reverse-mode autodiff ok)."""
  def _iter(val):
    next_val = body_fun(val)
    next_cond = cond_fun(next_val)
    return next_val, next_cond

  def _fun(tup, it):
    val, cond = tup
    # When cond is met, we start doing no-ops.
    # return torch.lax.cond(cond, _iter, lambda x: (x, False), val), it
    return condfun(cond, _iter, lambda x: (x, False), val), it

  init = (init_val, cond_fun(init_val))
  # return torch.lax.scan(_fun, init, None, length=max_iter)[0][0]
  return scan(_fun, init, None, length=max_iter)[0][0]


def _update_constraint(m: Model, d: Data, ctx: _Context) -> _Context:
  """Updates constraint force and resulting cost given latst solver iteration.

  Corresponds to CGupdateConstraint in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest qacc and smooth terms
    ctx: current solver context

  Returns:
    context with new constraint force and costs
  """
  del m

  # TODO(robotics-team): add friction constraints

  # only count active constraints
  # active = (ctx.Jaref < 0).at[:d.ne + d.nf].set(True)
  active = (ctx.Jaref < 0)
  active[:d.ne + d.nf] = True

  efc_force = d.efc_D * -ctx.Jaref * active
  qfrc_constraint = d.efc_J.T @ efc_force
  gauss = 0.5 * torch.dot(ctx.Ma - d.qfrc_smooth, ctx.qacc - d.qacc_smooth)
  cost = 0.5 * torch.sum(d.efc_D * ctx.Jaref * ctx.Jaref * active) + gauss

  ctx = ctx.replace(
      qfrc_constraint=qfrc_constraint,
      gauss=gauss,
      cost=cost,
      prev_cost=ctx.cost,
      efc_force=efc_force,
  )

  return ctx


def _update_gradient(m: Model, d: Data, ctx: _Context) -> _Context:
  """Updates grad and M / grad given latest solver iteration.

  Corresponds to CGupdateGradient in mujoco/src/engine/engine_solver.c

  Args:
    m: model defining constraints
    d: data which contains latest smooth terms
    ctx: current solver context

  Returns:
    context with new grad and M / grad
  Raises:
    NotImplementedError: for unsupported solver type
  """

  grad = ctx.Ma - d.qfrc_smooth - ctx.qfrc_constraint

  if m.opt.solver == SolverType.CG:
    mgrad = smooth.solve_m(m, d, grad)
  elif m.opt.solver == SolverType.NEWTON:
    # active = (ctx.Jaref < 0).at[:d.ne + d.nf].set(True)
    active = (ctx.Jaref < 0)
    active[:d.ne + d.nf] = True
    h = (d.efc_J.T * d.efc_D * active) @ d.efc_J
    h = smooth.dense_m(m, d) + h
    # h_ = torch.scipy.linalg.cho_factor(h)
    h_ = torch.linalg.cholesky(h, upper=True)
    # mgrad = torch.scipy.linalg.cho_solve(h_, grad)
    mgrad = torch.linalg.solve_triangular(h_, grad.unsqueeze(-1), upper=True).squeeze(-1)
  else:
    raise NotImplementedError(f"unsupported solver type: {m.opt.solver}")

  ctx = ctx.replace(grad=grad, Mgrad=mgrad)

  return ctx


def _rescale(m: Model, value: torch.Tensor) -> torch.Tensor:
  return value / (m.stat.meaninertia * max(1, m.nv))


def _linesearch(m: Model, d: Data, ctx: _Context) -> _Context:
  """Performs a zoom linesearch to find optimal search step size.

  Args:
    m: model defining search options and other needed terms
    d: data with inertia matrix and other needed terms
    ctx: current solver context

  Returns:
    updated context with new qacc, Ma, Jaref
  """
  from torch._C._functorch import is_batchedtensor, _remove_batch_dim

  smag = math.norm(ctx.search) * m.stat.meaninertia * max(1, m.nv)
  gtol = m.opt.tolerance * m.opt.ls_tolerance * smag

  # compute Mv, Jv
  mv = smooth.mul_m(m, d, ctx.search) if ctx.M is None else ctx.M @ ctx.search
  jv = d.efc_J @ ctx.search

  # prepare quadratics
  quad_gauss = torch.stack((
    ctx.gauss,
    torch.dot(ctx.search, ctx.Ma) - torch.dot(ctx.search, d.qfrc_smooth),
    0.5 * torch.dot(ctx.search, mv),
  ))
  quad = torch.stack((0.5 * ctx.Jaref * ctx.Jaref, jv * ctx.Jaref, 0.5 * jv * jv))
  quad = (quad * d.efc_D).T

  point_fn = lambda alpha: _LSPoint.create(d, ctx, alpha, jv, quad, quad_gauss)

  def cond(ctx: _LSContext) -> torch.Tensor:
    done = ctx.ls_iter >= m.opt.ls_iterations
    done = done | (~ctx.swap)  # if we did not adjust the interval
    done = done | ((ctx.lo.deriv_0 < 0) & (ctx.lo.deriv_0 > -gtol))
    done = done | ((ctx.hi.deriv_0 > 0) & (ctx.hi.deriv_0 < gtol))

    while is_batchedtensor(done):
      done = _remove_batch_dim(done, 1, 0, 0)
    return not done.all().item()

  def body(ctx: _LSContext) -> _LSContext:
    # always compute new bracket boundaries and a midpoint
    lo, hi = ctx.lo, ctx.hi
    lo_next = point_fn(lo.alpha - lo.deriv_0 / lo.deriv_1)
    hi_next = point_fn(hi.alpha - hi.deriv_0 / hi.deriv_1)
    mid = point_fn(0.5 * (lo.alpha + hi.alpha))

    # we swap lo/hi if:
    # 1) they are not correctly at a bracket boundary (e.g. lo.deriv_0 > 0), OR
    # 2) if moving to next or mid narrows the bracket
    swap_lo_next = (lo.deriv_0 > 0) | (lo.deriv_0 < lo_next.deriv_0)
    lo = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_lo_next, y, x), lo, lo_next)
    swap_lo_mid = (mid.deriv_0 < 0) & (lo.deriv_0 < mid.deriv_0)
    lo = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_lo_mid, y, x), lo, mid)

    swap_hi_next = (hi.deriv_0 < 0) | (hi.deriv_0 > hi_next.deriv_0)
    hi = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_hi_next, y, x), hi, hi_next)
    swap_hi_mid = (mid.deriv_0 > 0) & (hi.deriv_0 > mid.deriv_0)
    hi = torch.utils._pytree.tree_map(lambda x, y: torch.where(swap_hi_mid, y, x), hi, mid)

    swap = swap_lo_next | swap_lo_mid | swap_hi_next | swap_hi_mid

    ctx = ctx.replace(lo=lo, hi=hi, swap=swap, ls_iter=ctx.ls_iter + 1)

    return ctx

  # initialize interval
  p0 = point_fn(torch.tensor(0.0))
  lo = point_fn(p0.alpha - p0.deriv_0 / p0.deriv_1)
  lesser_fn = lambda x, y: torch.where(lo.deriv_0 < p0.deriv_0, x, y)
  hi = torch.utils._pytree.tree_map(lesser_fn, p0, lo)
  lo = torch.utils._pytree.tree_map(lesser_fn, lo, p0)
  ls_ctx = _LSContext(lo=lo, hi=hi, swap=torch.tensor(True), ls_iter=0)
  ls_ctx = _while_loop_scan(cond, body, ls_ctx, m.opt.ls_iterations)

  # move to new solution if improved
  lo, hi = ls_ctx.lo, ls_ctx.hi
  improved = (lo.cost < p0.cost) | (hi.cost < p0.cost)
  alpha = torch.where(lo.cost < hi.cost, lo.alpha, hi.alpha)
  qacc = ctx.qacc + improved * ctx.search * alpha
  ma = ctx.Ma + improved * mv * alpha
  jaref = ctx.Jaref + improved * jv * alpha

  ctx = ctx.replace(qacc=qacc, Ma=ma, Jaref=jaref)

  return ctx


def solve(m: Model, d: Data) -> Data:
  """Finds forces that satisfy constraints using conjugate gradient descent."""
  from torch._C._functorch import is_batchedtensor, _remove_batch_dim

  def cond(ctx: _Context) -> torch.Tensor:
    improvement = _rescale(m, ctx.prev_cost - ctx.cost)
    gradient = _rescale(m, torch.norm(ctx.grad))

    done = ctx.solver_niter >= m.opt.iterations
    done = done | (improvement < m.opt.tolerance)
    done = done | (gradient < m.opt.tolerance)
    while is_batchedtensor(done):
      done = _remove_batch_dim(done, 1, 0, 0)
    return not done.all().item()

  def body(ctx: _Context) -> _Context:
    ctx = _linesearch(m, d, ctx)
    prev_grad, prev_Mgrad = ctx.grad, ctx.Mgrad  # pylint: disable=invalid-name
    ctx = _update_constraint(m, d, ctx)
    ctx = _update_gradient(m, d, ctx)

    # polak-ribiere:
    beta = torch.dot(ctx.grad, ctx.Mgrad - prev_Mgrad)
    beta = beta / torch.clamp_min(torch.dot(prev_grad, prev_Mgrad), mujoco.mjMINVAL)
    # beta = beta / torch.maximum(mujoco.mjMINVAL, torch.dot(prev_grad, prev_Mgrad))
    beta = torch.clamp_min(beta, 0)
    # beta = torch.maximum(0, beta)
    search = -ctx.Mgrad + beta * ctx.search
    ctx = ctx.replace(search=search, solver_niter=ctx.solver_niter + 1)

    return ctx

  # warmstart:
  qacc = d.qacc_smooth
  if not m.opt.disableflags & DisableBit.WARMSTART:
    warm = _Context.create(m, d.replace(qacc=d.qacc_warmstart), grad=False)
    smth = _Context.create(m, d.replace(qacc=d.qacc_smooth), grad=False)
    qacc = torch.where(warm.cost < smth.cost, d.qacc_warmstart, d.qacc_smooth)
  d = d.replace(qacc=qacc)

  ctx = _Context.create(m, d)
  if m.opt.iterations == 1:
    ctx = body(ctx)
  else:
    while cond(ctx):
      ctx = body(ctx)
    # ctx = torch.lax.while_loop(cond, body, ctx)

  d = d.replace(
      qacc_warmstart=ctx.qacc,
      qacc=ctx.qacc,
      qfrc_constraint=ctx.qfrc_constraint,
      efc_force=ctx.efc_force,
  )

  return d

def scan(f, init, xs, length=None):
  if xs is None:
    xs = [None] * length
  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys.append(y)
  return carry, torch.stack(ys) if all(y is not None for y in ys) else None

def condfun(pred, true_fun, false_fun, *operands):
  if pred:
    return true_fun(*operands)
  else:
    return false_fun(*operands)
