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
"""Collision functions for shapes represented as signed distance functions (SDF).

A signed distance function at a given point in space is the shortest distance to
a surface. This enables to define a geometry implicitly and exactly.

See https://iquilezles.org/articles/distfunctions/ for a list of analytic SDFs.
"""

import functools
from typing import Callable, Tuple

import torch
from torch._higher_order_ops.scan import scan as torch_scan
from mujoco_torch._src import math
# pylint: disable=g-importing-member
from mujoco_torch._src.collision_types import Collision as Contact
from mujoco_torch._src.collision_types import GeomInfo
# pylint: enable=g-importing-member

# the SDF function takes position in, and returns a distance or objective
SDFFn = Callable[[torch.Tensor], torch.Tensor]


def _plane(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  del size
  return pos[2]


def _sphere(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  return math.norm(pos) - size[0]


def _capsule(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  pa = -size[1] * torch.tensor([0, 0, 1], dtype=pos.dtype, device=pos.device)
  pb = size[1] * torch.tensor([0, 0, 1], dtype=pos.dtype, device=pos.device)
  ab = pb - pa
  ap = pos - pa
  denom = ab.dot(ab)
  denom = torch.where(torch.abs(denom) < 1e-12, 1e-12 * math.sign(denom), denom)
  t = ab.dot(ap) / denom
  t = torch.clamp(t, 0, 1)
  c = pa + t * ab
  return math.norm(pos - c) - size[0]


def _ellipsoid(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  k0 = math.norm(pos / size)
  k1 = math.norm(pos / (size * size))
  return k0 * (k0 - 1.0) / (k1 + (k1 == 0.0) * 1e-12)


def _cylinder_grad(x: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  """Gradient of the cylinder SDF wrt query point and singularities removed."""
  c = torch.sqrt(x[0] * x[0] + x[1] * x[1])
  e = torch.abs(x[2])
  a = torch.stack([c - size[0], e - size[1]])
  b = torch.stack([
      torch.maximum(a[0], torch.tensor(0.0, dtype=x.dtype, device=x.device)),
      torch.maximum(a[1], torch.tensor(0.0, dtype=x.dtype, device=x.device)),
  ])
  j = torch.argmax(a)
  bnorm = torch.sqrt(b[0] * b[0] + b[1] * b[1])
  bnorm = bnorm + (torch.allclose(bnorm, torch.zeros_like(bnorm)) * 1e-12)
  grada = torch.stack([
      x[0] / (c + (torch.allclose(c, torch.zeros_like(c)) * 1e-12)),
      x[1] / (c + (torch.allclose(c, torch.zeros_like(c)) * 1e-12)),
      x[2] / (e + (torch.allclose(e, torch.zeros_like(e)) * 1e-12)),
  ])
  gradm = torch.stack([
      torch.stack([grada[0], grada[1], torch.tensor(0.0, dtype=x.dtype, device=x.device)]),
      torch.stack([torch.tensor(0.0, dtype=x.dtype, device=x.device), torch.tensor(0.0, dtype=x.dtype, device=x.device), grada[2]]),
  ])
  b_idx = torch.stack([b[0], b[0], b[1]])
  gradb = grada * b_idx / bnorm
  return torch.where(a[j] < 0, gradm[j], gradb)


class _CylinderSDF(torch.autograd.Function):
  """Custom autograd for cylinder SDF with correct gradient at singularities."""

  @staticmethod
  def forward(ctx, pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
    ctx.save_for_backward(pos, size)
    a0 = torch.sqrt(pos[0] * pos[0] + pos[1] * pos[1]) - size[0]
    a1 = torch.abs(pos[2]) - size[1]
    b0 = torch.maximum(a0, torch.tensor(0.0, dtype=pos.dtype, device=pos.device))
    b1 = torch.maximum(a1, torch.tensor(0.0, dtype=pos.dtype, device=pos.device))
    return torch.minimum(
        torch.maximum(a0, a1), torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
    ) + torch.sqrt(b0 * b0 + b1 * b1)

  @staticmethod
  def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
    pos, size = ctx.saved_tensors
    grad_x = _cylinder_grad(pos, size)
    return grad_output * grad_x, None


def _cylinder(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  return _CylinderSDF.apply(pos, size)


def _from_to(
    f: SDFFn,
    from_pos: torch.Tensor,
    from_mat: torch.Tensor,
    to_pos: torch.Tensor,
    to_mat: torch.Tensor,
) -> SDFFn:
  relmat = math.matmul_unroll(to_mat.T, from_mat)
  relpos = to_mat.T @ (from_pos - to_pos)

  def wrapped(p: torch.Tensor) -> torch.Tensor:
    return f(relmat @ p + relpos)

  return wrapped


def _intersect(d1: SDFFn, d2: SDFFn) -> SDFFn:
  return lambda p: torch.maximum(d1(p), d2(p))


def _clearance(d1: SDFFn, d2: SDFFn) -> SDFFn:
  def fn(p: torch.Tensor) -> torch.Tensor:
    return (d1(p) + d2(p) + torch.abs(_intersect(d1, d2)(p))).squeeze()
  return fn


def _gradient_step(
    objective: SDFFn, state: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Performs a step of gradient descent."""
  dist, x = state
  amin = 1e-4
  amax = 2.0
  nlinesearch = 10
  alpha = torch.logspace(
      torch.log10(torch.tensor(amin, dtype=x.dtype, device=x.device)),
      torch.log10(torch.tensor(amax, dtype=x.dtype, device=x.device)),
      nlinesearch,
  )
  x_grad = x.detach().clone().requires_grad_(True)
  grad = torch.autograd.grad(objective(x_grad), x_grad)[0]
  candidates = x.unsqueeze(0) - alpha.unsqueeze(1) * grad.unsqueeze(0)
  values = torch.stack([objective(candidates[i]) for i in range(nlinesearch)])
  idx = torch.argmin(values)
  return values[idx], candidates[idx]


def _gradient_descent(
    objective: SDFFn,
    x: torch.Tensor,
    niter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Performs gradient descent with backtracking line search."""
  dist = torch.tensor(1e10, dtype=x.dtype, device=x.device)
  init = (dist, x)

  def combine_fn(carry, _unused):
    new_carry = _gradient_step(objective, carry)
    return new_carry, torch.zeros(())

  xs = torch.arange(niter, dtype=x.dtype, device=x.device)
  (dist, x), _ = torch_scan(combine_fn, init, xs, dim=0)

  return dist, x


def _optim(
    d1: SDFFn,
    d2: SDFFn,
    info1: GeomInfo,
    info2: GeomInfo,
    x0: torch.Tensor,
) -> Contact:
  """Optimizes the clearance function."""
  d1 = functools.partial(d1, size=info1.geom_size)
  d1 = _from_to(d1, info2.pos, info2.mat, info1.pos, info1.mat)
  d2 = functools.partial(d2, size=info2.geom_size)
  x0 = info2.mat.T @ (x0 - info2.pos)
  fn = _clearance(d1, d2)
  _, pos = _gradient_descent(fn, x0, 10)
  dist_val = d1(pos) + d2(pos)
  pos_grad = pos.detach().clone().requires_grad_(True)
  n = torch.autograd.grad(d1(pos_grad), pos_grad)[0]
  pos_grad2 = pos.detach().clone().requires_grad_(True)
  n = n - torch.autograd.grad(d2(pos_grad2), pos_grad2)[0]
  pos = info2.mat @ pos + info2.pos
  n = info2.mat @ n
  return dist_val, pos, math.make_frame(n)


def sphere_ellipsoid(s: GeomInfo, e: GeomInfo) -> Contact:
  """Calculates contact between a sphere and an ellipsoid."""
  x0 = 0.5 * (s.pos + e.pos)
  return _optim(_sphere, _ellipsoid, s, e, x0)


def sphere_cylinder(s: GeomInfo, c: GeomInfo) -> Contact:
  """Calculates contact between a sphere and a cylinder."""
  x0 = 0.5 * (s.pos + c.pos)
  return _optim(_sphere, _cylinder, s, c, x0)


def capsule_ellipsoid(c: GeomInfo, e: GeomInfo) -> Contact:
  """Calculates contact between a capsule and an ellipsoid."""
  x0 = 0.5 * (c.pos + e.pos)
  return _optim(_capsule, _ellipsoid, c, e, x0)


def capsule_cylinder(ca: GeomInfo, cy: GeomInfo) -> Contact:
  """Calculates contact between a capsule and a cylinder."""
  mid = 0.5 * (ca.pos + cy.pos)
  vec = ca.mat[:, 2] * ca.geom_size[1]
  x0 = torch.stack([mid - vec, mid + vec])
  return torch.vmap(functools.partial(_optim, _capsule, _cylinder, ca, cy))(x0)


def ellipsoid_ellipsoid(e1: GeomInfo, e2: GeomInfo) -> Contact:
  """Calculates contact between two ellipsoids."""
  x0 = 0.5 * (e1.pos + e2.pos)
  return _optim(_ellipsoid, _ellipsoid, e1, e2, x0)


def ellipsoid_cylinder(e: GeomInfo, c: GeomInfo) -> Contact:
  """Calculates contact between an ellipsoid and a cylinder."""
  x0 = 0.5 * (e.pos + c.pos)
  return _optim(_ellipsoid, _cylinder, e, c, x0)


def cylinder_cylinder(c1: GeomInfo, c2: GeomInfo) -> Contact:
  """Calculates contact between a cylinder and a cylinder."""
  basis = math.make_frame(c2.pos - c1.pos)
  mid = 0.5 * (c1.pos + c2.pos)
  r = torch.maximum(c1.geom_size[0], c2.geom_size[0])
  x0 = torch.stack([
      mid + r * basis[1],
      mid + r * basis[2],
      mid - r * basis[1],
      mid - r * basis[2],
  ])
  return torch.vmap(functools.partial(_optim, _cylinder, _cylinder, c1, c2))(x0)


# store ncon as function attributes
sphere_ellipsoid.ncon = 1
sphere_cylinder.ncon = 1
capsule_ellipsoid.ncon = 1
capsule_cylinder.ncon = 2
ellipsoid_ellipsoid.ncon = 1
ellipsoid_cylinder.ncon = 1
cylinder_cylinder.ncon = 4
