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
"""Functions for ray interesection testing."""

from typing import Sequence, Tuple

import mujoco
from mujoco_torch._src import math
# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import GeomType
from mujoco_torch._src.types import Model
# pylint: enable=g-importing-member
import numpy as np


def _ray_quad(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
  det = b * b - a * c
  det_2 = torch.sqrt(det)

  x0, x1 = (-b - det_2) / a, (-b + det_2) / a
  x0 = torch.where((det < mujoco.mjMINVAL) | (x0 < 0), float("inf"), x0)
  x1 = torch.where((det < mujoco.mjMINVAL) | (x1 < 0), float("inf"), x1)

  return x0, x1


def _ray_plane(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
  """Returns the distance at which a ray intersects with a plane."""
  x = -pnt[2] / vec[2]

  valid = vec[2] <= -mujoco.mjMINVAL  # z-vec pointing towards front face
  valid &= x >= 0
  # only within rendered rectangle
  p = pnt[0:2] + x * vec[0:2]
  valid &= torch.all((size[0:2] <= 0) | (torch.abs(p) <= size[0:2]))

  return torch.where(valid, x, float("inf"))


def _ray_sphere(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
  """Returns the distance at which a ray intersects with a sphere."""
  x0, x1 = _ray_quad(vec @ vec, vec @ pnt, pnt @ pnt - size[0] * size[0])
  x = torch.where(torch.isinf(x0), x1, x0)

  return x


def _ray_capsule(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
  """Returns the distance at which a ray intersects with a capsule."""

  # cylinder round side: (x*lvec+lpnt)'*(x*lvec+lpnt) = size[0]*size[0]
  a = vec[0:2] @ vec[0:2]
  b = vec[0:2] @ pnt[0:2]
  c = pnt[0:2] @ pnt[0:2] - size[0] * size[0]

  # solve a*x^2 + 2*b*x + c = 0
  x0, x1 = _ray_quad(a, b, c)
  x = torch.where(torch.isinf(x0), x1, x0)

  # make sure round solution is between flat sides
  x = torch.where(torch.abs(pnt[2] + x * vec[2]) <= size[1], x, float("inf"))

  # top cap
  dif = pnt - torch.tensor([0, 0, size[1]])
  x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])
  # accept only top half of sphere
  x = torch.where((pnt[2] + x0 * vec[2] >= size[1]) & (x0 < x), x0, x)
  x = torch.where((pnt[2] + x1 * vec[2] >= size[1]) & (x1 < x), x1, x)

  # bottom cap
  dif = pnt + torch.tensor([0, 0, size[1]])
  x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])

  # accept only bottom half of sphere
  x = torch.where((pnt[2] + x0 * vec[2] <= -size[1]) & (x0 < x), x0, x)
  x = torch.where((pnt[2] + x1 * vec[2] <= -size[1]) & (x1 < x), x1, x)

  return x


def _ray_box(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
  """Returns the distance at which a ray intersects with a box."""

  iface = torch.tensor([(1, 2), (0, 2), (0, 1), (1, 2), (0, 2), (0, 1)])

  # side +1, -1
  # solution of pnt[i] + x * vec[i] = side * size[i]
  x = concatenate([(size - pnt) / vec, (-size - pnt) / vec])

  # intersection with face
  p0 = pnt[iface[:, 0]] + x * vec[iface[:, 0]]
  p1 = pnt[iface[:, 1]] + x * vec[iface[:, 1]]
  valid = torch.abs(p0) <= size[iface[:, 0]]
  valid &= torch.abs(p1) <= size[iface[:, 1]]

  return torch.min(torch.where(valid, x, float("inf")))


def _ray_triangle(
    vert: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
) -> torch.Tensor:
  """Returns the distance at which a ray intersects with a triangle."""
  # project difference vectors in ray normal plane
  planar = torch.dot(torch.tensor([b0, b1]), (vert - pnt).T)

  # determine if origin is inside planar projection of triangle
  # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
  A = torch.tensor(  # pylint: disable=invalid-name
      [planar[:, 0] - planar[:, 2], planar[:, 1] - planar[:, 2]]
  ).T.flatten()
  b = -planar[:, 2]
  det = A[0] * A[3] - A[1] * A[2]
  valid = torch.abs(det) >= mujoco.mjMINVAL

  t0 = (A[3] * b[0] - A[1] * b[1]) / det
  t1 = (-A[2] * b[0] + A[0] * b[1]) / det
  valid &= (t0 >= 0) & (t1 >= 0) & (t0 + t1 <= 1)

  # intersect ray with plane of triangle
  nrm = torch.linalg.cross(vert[0] - vert[2], vert[1] - vert[2])
  denom = torch.dot(vec, nrm)
  valid &= torch.abs(denom) >= mujoco.mjMINVAL

  dist = torch.where(valid, -torch.dot(pnt - vert[2], nrm) / denom, float("inf"))

  return dist


def _ray_mesh(
    m: Model,
    geom_id: np.ndarray,
    unused_size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns the best distance and geom_id for ray mesh intersections."""
  data_id = m.geom_dataid[geom_id]

  ray_basis = lambda x: math.orthogonals(math.normalize(x))
  b0, b1 = torch.vmap(ray_basis)(vec)

  faceadr = np.append(m.mesh_faceadr, m.nmeshface)
  vertadr = np.append(m.mesh_vertadr, m.nmeshvert)

  dists = []
  for i, id_ in enumerate(data_id):
    face = m.mesh_face[faceadr[id_] : faceadr[id_ + 1]]
    vert = m.mesh_vert[vertadr[id_] : vertadr[id_ + 1]]
    dist = torch.vmap(_ray_triangle, in_axes=(0, None, None, None, None))(
        vert[face], pnt[i], vec[i], b0[i], b1[i]
    )
    dists.append(dist)

  # map the triangle id to data id
  tri_id = np.append(0, (faceadr[data_id + 1] - faceadr[data_id]).cumsum())
  tri_data_id = np.zeros(tri_id[-1], dtype=np.int32)
  tri_data_id[tri_id[:-1]] = 1
  tri_data_id = tri_data_id.cumsum() - 1

  dists = concatenate(dists)
  min_id = torch.argmin(dists)
  # Grab the best distance amongst all meshes, bypassing the argmin in `ray`.
  # This avoids having to compute the best distance per mesh.
  dist = dists[min_id, None]
  id_ = torch.tensor(geom_id)[torch.tensor(tri_data_id)[min_id], None]

  return dist, id_


_RAY_FUNC = {
    GeomType.PLANE: _ray_plane,
    GeomType.SPHERE: _ray_sphere,
    GeomType.CAPSULE: _ray_capsule,
    GeomType.BOX: _ray_box,
    GeomType.MESH: _ray_mesh,
}


def ray(
    m: Model,
    d: Data,
    pnt: torch.Tensor,
    vec: torch.Tensor,
    geomgroup: Sequence[int] = (),
    flg_static: bool = True,
    bodyexclude: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns the geom id and distance at which a ray intersects with a geom.

  Args:
    m: MJX model
    d: MJX data
    pnt: ray origin point (3,)
    vec: ray direction    (3,)
    geomgroup: group inclusion/exclusion mask, or empty to ignore
    flg_static: if True, allows rays to intersect with static geoms
    bodyexclude: ignore geoms on specified body id

  Returns:
    dist: distance from ray origin to geom surface (or -1.0 for no intersection)
    id: id of intersected geom (or -1 for no intersection)
  """

  dists, ids = [], []
  geom_filter = m.geom_bodyid != bodyexclude
  geom_filter &= (m.geom_matid != -1) | (m.geom_rgba[:, 3] != 0)
  geom_filter &= (m.geom_matid == -1) | (m.mat_rgba[m.geom_matid, 3] != 0)
  geom_filter &= flg_static | (m.body_weldid[m.geom_bodyid] != 0)
  if geomgroup:
    geomgroup = np.array(geomgroup, dtype=bool)
    geom_filter &= geomgroup[np.clip(m.geom_group, 0, mujoco.mjNGROUP)]

  # map ray to local geom frames
  geom_pnts = torch.vmap(lambda x, y: x.T @ (pnt - y))(d.geom_xmat, d.geom_xpos)
  geom_vecs = torch.vmap(lambda x: x.T @ vec)(d.geom_xmat)

  for geom_type, fn in _RAY_FUNC.items():
    id_, = torch.nonzero(geom_filter & (m.geom_type == geom_type))

    if id_.size == 0:
      continue

    args = m.geom_size[id_], geom_pnts[id_], geom_vecs[id_]

    if geom_type == GeomType.MESH:
      dist, id_ = fn(m, id_, *args)
    else:
      dist = torch.vmap(fn)(*args)

    dists, ids = dists + [dist], ids + [id_]

  if not ids:
    return torch.tensor(-1), torch.tensor(-1.0)

  dists = concatenate(dists)
  ids = concatenate(ids)
  min_id = torch.argmin(dists)
  dist = torch.where(torch.isinf(dists[min_id]), -1, dists[min_id])
  id_ = torch.where(torch.isinf(dists[min_id]), -1, ids[min_id])

  return dist, id_
