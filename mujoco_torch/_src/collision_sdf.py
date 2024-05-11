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
from typing import Callable
from typing import Tuple

import torch
from mujoco_torch._src import math
# pylint: disable=g-importing-member
from mujoco_torch._src.collision_base import Contact
from mujoco_torch._src.collision_base import GeomInfo
from mujoco_torch._src.dataclasses import PyTreeNode
# pylint: enable=g-importing-member


# the objective function, inputs: pos (that we optimize for) and size (known)
SDFFn = Callable[[torch.Tensor], torch.Tensor]


def _plane(pos: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
  del size
  return pos[2]


def _sphere(pos: torch.Tensor, size: torch.Tensor):
  return math.norm(pos) - size[0]


def _capsule(pos: torch.Tensor, size: torch.Tensor):
  pa = -size[1] * torch.tensor([0, 0, 1])
  pb = size[1] * torch.tensor([0, 0, 1])
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
  k1 = math.norm(pos / (size*size))
  return k0 * (k0 - 1.0) / (k1 + (k1 == 0.0) * 1e-12)


def _to_local(f: SDFFn, pos: torch.Tensor, mat: torch.Tensor)-> SDFFn:
  return lambda p: f(mat.T @ (p - pos))


def _intersect(d1: SDFFn, d2: SDFFn) -> SDFFn:
  return lambda p: torch.maximum(d1(p), d2(p))


def _clearance(d1: SDFFn, d2: SDFFn) -> SDFFn:
  return lambda p: (d1(p) + d2(p) + torch.abs(_intersect(d1, d2)(p))).squeeze()


class GradientState(PyTreeNode):
  dist: torch.Tensor
  x: torch.Tensor


def _gradient_step(objective: SDFFn, state: GradientState) -> GradientState:
  """Performs a step of gradient descent."""
  # TODO: find better parameters
  amin = 1e-4  # minimum value for line search factor scaling the gradient
  amax = 2.  # maximum value for line search factor scaling the gradient
  nlinesearch = 10  # line search points
  grad = jax.grad(objective)(state.x)
  alpha = torch_geomspace(amin, amax, nlinesearch).reshape(nlinesearch, -1)
  candidates = state.x - alpha * grad.reshape(-1, 3)
  values = torch.vmap(objective)(candidates)
  idx = torch.argmin(values)
  return state.replace(x=candidates[idx], dist=values[idx])


def _gradient_descent(
    objective: SDFFn,
    x: torch.Tensor,
    niter: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Performs gradient descent with backtracking line search."""
  state = GradientState(
      dist=1e10,
      x=x,
  )

  state, _ = jax.lax.scan(
      lambda s, _: (_gradient_step(objective, s), None), state, (), length=niter
  )
  return state.dist, state.x


def _optim(
    d1, d2, info1: GeomInfo, info2: GeomInfo
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Optimizes the clearance function."""
  d1 = functools.partial(d1, size=info1.size)
  d1 = _to_local(d1, info1.pos, info1.mat)
  d2 = functools.partial(d2, size=info2.size)
  d2 = _to_local(d2, info2.pos, info2.mat)
  fn = _clearance(d1, d2)
  _, pos = _gradient_descent(fn, 0.5 * (info1.pos + info2.pos), 10)
  dist = d1(pos) + d2(pos)
  n = jax.grad(d1)(pos)
  return pos, dist, n


def capsule_ellipsoid(c: GeomInfo, e: GeomInfo) -> Contact:
  """"Calculates contact between a capsule and an ellipsoid."""
  pos, dist, n = _optim(_capsule, _ellipsoid, c, e)
  return torch.utils._pytree.tree_map(
      lambda x: x.unsqueeze(0), (dist, pos, math.make_frame(n))
  )


def ellipsoid_ellipsoid(e1: GeomInfo, e2: GeomInfo) -> Contact:
  """"Calculates contact between two ellipsoids."""
  pos, dist, n = _optim(_ellipsoid, _ellipsoid, e1, e2)
  return torch.utils._pytree.tree_map(
      lambda x: x.unsqueeze(0), (dist, pos, math.make_frame(n))
  )


# store ncon as function attributes
capsule_ellipsoid.ncon = 1
ellipsoid_ellipsoid.ncon = 1
