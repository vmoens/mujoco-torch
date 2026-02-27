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
"""Functions for ray intersection testing."""

from collections.abc import Sequence

import mujoco
import numpy as np
import torch

from mujoco_torch._src import math
from mujoco_torch._src.types import Data, GeomType, Model


def _ray_quad(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns two solutions for quadratic: a*x^2 + 2*b*x + c = 0."""
    det = b * b - a * c
    det_2 = torch.sqrt(det)

    x0 = math.safe_div(-b - det_2, a)
    x1 = math.safe_div(-b + det_2, a)
    x0 = torch.where((det < mujoco.mjMINVAL) | (x0 < 0), torch.full((), torch.inf, dtype=x0.dtype, device=x0.device), x0)
    x1 = torch.where((det < mujoco.mjMINVAL) | (x1 < 0), torch.full((), torch.inf, dtype=x1.dtype, device=x1.device), x1)

    return x0, x1


def _ray_plane(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Returns the distance at which a ray intersects with a plane."""
    x = -math.safe_div(pnt[2], vec[2])

    valid = vec[2] <= -mujoco.mjMINVAL  # z-vec pointing towards front face
    valid = valid & (x >= 0)
    # only within rendered rectangle
    p = pnt[0:2] + x * vec[0:2]
    valid = valid & torch.all((size[0:2] <= 0) | (torch.abs(p) <= size[0:2]))

    return torch.where(valid, x, torch.full((), torch.inf, dtype=x.dtype, device=x.device))


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
    x = torch.where(
        torch.abs(pnt[2] + x * vec[2]) <= size[1], x, torch.full((), torch.inf, dtype=x.dtype, device=x.device)
    )

    # top cap
    z_offset = torch.stack([torch.zeros_like(size[1]), torch.zeros_like(size[1]), size[1]])
    dif = pnt - z_offset
    x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])
    x = torch.where((pnt[2] + x0 * vec[2] >= size[1]) & (x0 < x), x0, x)
    x = torch.where((pnt[2] + x1 * vec[2] >= size[1]) & (x1 < x), x1, x)

    # bottom cap
    dif = pnt + z_offset
    x0, x1 = _ray_quad(vec @ vec, vec @ dif, dif @ dif - size[0] * size[0])
    x = torch.where((pnt[2] + x0 * vec[2] <= -size[1]) & (x0 < x), x0, x)
    x = torch.where((pnt[2] + x1 * vec[2] <= -size[1]) & (x1 < x), x1, x)

    return x


def _ray_ellipsoid(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Returns the distance at which a ray intersects with an ellipsoid."""

    # invert size^2
    s = math.safe_div(1, torch.square(size))

    # (x*lvec+lpnt)' * diag(1/size^2) * (x*lvec+lpnt) = 1
    svec = s * vec
    a = svec @ vec
    b = svec @ pnt
    c = (s * pnt) @ pnt - 1

    # solve a*x^2 + 2*b*x + c = 0
    x0, x1 = _ray_quad(a, b, c)
    x = torch.where(torch.isinf(x0), x1, x0)

    return x


def _ray_box(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """Returns the distance at which a ray intersects with a box."""

    iface = torch.tensor(
        [[1, 2], [0, 2], [0, 1], [1, 2], [0, 2], [0, 1]],
        dtype=torch.long,
        device=pnt.device,
    )

    # side +1, -1
    # solution of pnt[i] + x * vec[i] = side * size[i]
    x = torch.cat(
        [
            math.safe_div(size - pnt, vec),
            -math.safe_div(size + pnt, vec),
        ]
    )

    # intersection with face
    p0 = pnt[iface[:, 0]] + x * vec[iface[:, 0]]
    p1 = pnt[iface[:, 1]] + x * vec[iface[:, 1]]
    valid = torch.abs(p0) <= size[iface[:, 0]]
    valid = valid & (torch.abs(p1) <= size[iface[:, 1]])
    valid = valid & (x >= 0)

    return torch.min(torch.where(valid, x, torch.full((), torch.inf, dtype=x.dtype, device=x.device)))


def _ray_triangle(
    vert: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """Returns the distance at which a ray intersects with a triangle."""
    # project difference vectors in ray normal plane
    planar = (vert - pnt) @ basis

    # determine if origin is inside planar projection of triangle
    # A = (p0-p2, p1-p2), b = -p2, solve A*t = b
    A = planar[0:2] - planar[2]  # pylint: disable=invalid-name
    b = -planar[2]
    det = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]

    t0 = math.safe_div(A[1, 1] * b[0] - A[1, 0] * b[1], det)
    t1 = math.safe_div(-A[0, 1] * b[0] + A[0, 0] * b[1], det)
    valid = (t0 >= 0) & (t1 >= 0) & (t0 + t1 <= 1)

    # intersect ray with plane of triangle
    nrm = math.cross(vert[0] - vert[2], vert[1] - vert[2])
    dist = math.safe_div(
        torch.dot(vert[2] - pnt, nrm),
        torch.dot(vec, nrm),
    )
    valid = valid & (dist >= 0)
    dist = torch.where(valid, dist, torch.full((), torch.inf, dtype=dist.dtype, device=dist.device))

    return dist


def _ray_mesh(
    m: Model,
    geom_id: np.ndarray,
    unused_size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the best distance and geom_id for ray mesh intersections."""
    data_id = m.geom_dataid[geom_id]

    def ray_basis(x):
        return torch.tensor(math.orthogonals(math.normalize(x)), dtype=x.dtype, device=x.device).T

    basis = torch.vmap(ray_basis)(vec)

    faceadr = np.append(m.mesh_faceadr, m.nmeshface)
    vertadr = np.append(m.mesh_vertadr, m.nmeshvert)

    dists_list, geom_ids_list = [], []
    for i, id_ in enumerate(data_id):
        face = m.mesh_face[faceadr[id_] : faceadr[id_ + 1]]
        vert = m.mesh_vert[vertadr[id_] : vertadr[id_ + 1]]
        vert = torch.tensor(vert[face], dtype=pnt.dtype, device=pnt.device)
        dist = torch.vmap(
            lambda v, p, vec, b: _ray_triangle(v, p, vec, b),
            in_dims=(0, None, None, None),
        )(vert, pnt[i], vec[i], basis[i])
        dists_list.append(dist)
        geom_ids_list.append(np.repeat(geom_id[i], dist.shape[0]))

    dists = torch.cat(dists_list)
    geom_ids_flat = np.concatenate(geom_ids_list)
    min_id = torch.argmin(dists)
    dist = dists[min_id].unsqueeze(0)
    id_ = torch.tensor(geom_ids_flat[min_id], dtype=torch.long, device=pnt.device).unsqueeze(0)

    return dist, id_


_RAY_FUNC = {
    GeomType.PLANE: _ray_plane,
    GeomType.SPHERE: _ray_sphere,
    GeomType.CAPSULE: _ray_capsule,
    GeomType.ELLIPSOID: _ray_ellipsoid,
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
    bodyexclude: Sequence[int] | int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the geom id and distance at which a ray intersects with a geom.

    Args:
      m: MJX model
      d: MJX data
      pnt: ray origin point (3,)
      vec: ray direction (3,)
      geomgroup: group inclusion/exclusion mask, or empty to ignore
      flg_static: if True, allows rays to intersect with static geoms
      bodyexclude: ignore geoms on specified body id or sequence of body ids

    Returns:
      Distance from ray origin to geom surface (or -1.0 for no intersection) and
      id of intersected geom (or -1 for no intersection)
    """

    dists, ids = [], []
    if not isinstance(bodyexclude, Sequence):
        bodyexclude = [bodyexclude]
    geom_filter = flg_static | (m.body_weldid[m.geom_bodyid] != 0)
    for bodyid in bodyexclude:
        geom_filter = geom_filter & (m.geom_bodyid != bodyid)
    if geomgroup:
        geomgroup = np.array(geomgroup, dtype=bool)
        geom_filter = geom_filter & geomgroup[np.clip(m.geom_group, 0, mujoco.mjNGROUP)]

    # map ray to local geom frames
    geom_pnts = torch.vmap(lambda x, y: x.T @ (pnt - y))(d.geom_xmat, d.geom_xpos)
    geom_vecs = torch.vmap(lambda x: x.T @ vec)(d.geom_xmat)

    geom_filter_dyn = (m.geom_matid != -1) | (m.geom_rgba[:, 3] != 0)
    geom_filter_dyn = geom_filter_dyn & ((m.geom_matid == -1) | (m.mat_rgba[m.geom_matid, 3] != 0))
    for geom_type, fn in _RAY_FUNC.items():
        (id_,) = np.nonzero(geom_filter & (m.geom_type == geom_type))

        if id_.size == 0:
            continue

        args = m.geom_size[id_], geom_pnts[id_], geom_vecs[id_]

        if geom_type == GeomType.MESH:
            dist, id_ = fn(m, id_, *args)
        else:
            dist = torch.vmap(fn)(*args)

        dist = torch.where(geom_filter_dyn[id_], dist, torch.full((), torch.inf, dtype=dist.dtype, device=dist.device))
        dists.append(dist)
        ids.append(id_)

    if not ids:
        device = pnt.device if isinstance(pnt, torch.Tensor) else None
        return torch.full((), -1, dtype=torch.long, device=device), torch.full((), -1.0, dtype=pnt.dtype, device=device)

    dists = torch.cat(dists)
    ids_cat = torch.cat([torch.tensor(x, dtype=torch.long, device=pnt.device) for x in ids])
    min_id = torch.argmin(dists)
    min_dist = dists.gather(0, min_id.unsqueeze(0)).squeeze(0)
    min_geom_id = ids_cat.gather(0, min_id.unsqueeze(0)).squeeze(0)
    dist = torch.where(torch.isinf(min_dist), torch.full((), -1.0, dtype=dists.dtype, device=dists.device), min_dist)
    id_ = torch.where(torch.isinf(min_dist), torch.full((), -1, dtype=torch.long, device=ids_cat.device), min_geom_id)

    return dist, id_


def ray_geom(
    size: torch.Tensor,
    pnt: torch.Tensor,
    vec: torch.Tensor,
    geomtype: GeomType,
) -> torch.Tensor:
    """Returns the distance at which a ray intersects with a primitive geom.

    Args:
      size: geom size (1,), (2,), or (3,)
      pnt: ray origin point (3,)
      vec: ray direction (3,)
      geomtype: type of geom

    Returns:
      dist: distance from ray origin to geom surface
    """
    return _RAY_FUNC[geomtype](size, pnt, vec)
