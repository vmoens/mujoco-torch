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
"""Heightfield collisions."""

import numpy as np
import torch
import torch.nn.functional as F

from mujoco_torch._src import math
from mujoco_torch._src import mesh as mesh_module
from mujoco_torch._src.math import _CachedConst
from mujoco_torch._src.collision_convex import (
    _clip_edge_to_planes,
    _manifold_points,
    _project_pt_onto_plane,
    _sat_hull_hull,
    _vmap_select,
    _vmap_take,
    _vmap_take_1d,
)
from mujoco_torch._src.collision_types import (
    Collision,
    ConvexInfo,
    GeomInfo,
    HFieldInfo,
)

_MANIFOLD_TOL = _CachedConst(1e-3)


def _sphere_prism(sphere: GeomInfo, prism: ConvexInfo) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sphere vs prism collision. Returns (dist, pos, n)."""
    faces = prism.face
    normals = prism.face_normal

    sphere_pos = prism.mat.T @ (sphere.pos - prism.pos)

    @torch.vmap
    def get_support(face, normal):
        pos = sphere_pos - normal * sphere.geom_size[0]
        return torch.dot(pos - face[0], normal)

    support = get_support(faces, normals)
    support = torch.where(
        support >= 0,
        torch.full((), -1e12, dtype=support.dtype, device=support.device),
        support,
    )
    best_idx = support.argmax()
    face = _vmap_select(faces, best_idx)
    normal = _vmap_select(normals, best_idx)

    pt = _project_pt_onto_plane(sphere_pos, face[0], normal)
    edge_p0 = torch.roll(face, 1, dims=0)
    edge_p1 = face
    edge_normals = torch.vmap(math.cross, (0, None))(edge_p1 - edge_p0, normal)
    edge_dist = torch.vmap(lambda pp, pn: torch.dot(pt - pp, pn))(edge_p0, edge_normals)
    inside = torch.all(edge_dist <= 0)

    degenerate = torch.all(edge_normals == 0, dim=1)
    behind = edge_dist < 0.0
    edge_dist = torch.where(
        degenerate | behind,
        torch.full((), 1e12, dtype=edge_dist.dtype, device=edge_dist.device),
        edge_dist,
    )
    idx = edge_dist.argmin()
    edge_pt = math.closest_segment_point(_vmap_select(edge_p0, idx), _vmap_select(edge_p1, idx), pt)
    pt = torch.where(inside, pt, edge_pt)

    n, d = math.normalize_with_norm(pt - sphere_pos)
    spt = sphere_pos + n * sphere.geom_size[0]
    dist = d - sphere.geom_size[0]
    pos = (pt + spt) * 0.5

    n = prism.mat @ n
    pos = prism.mat @ pos + prism.pos
    return dist, pos, n


def _capsule_prism(cap: GeomInfo, prism: ConvexInfo) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capsule vs prism. Returns (dist(2,), pos(2,3), n(2,3))."""
    faces = prism.face
    normals = prism.face_normal

    cap_pos = prism.mat.T @ (cap.pos - prism.pos)
    axis, length = cap.mat[:, 2], cap.geom_size[1]
    axis = prism.mat.T @ axis
    seg = axis * length
    cap_pts = torch.stack([cap_pos - seg, cap_pos + seg])

    @torch.vmap
    def get_support(face, normal):
        pts = cap_pts - normal * cap.geom_size[0]
        sup = torch.vmap(lambda x: torch.dot(x - face[0], normal))(pts)
        return sup.min()

    support = get_support(faces, normals)
    support = torch.where(
        support >= 0,
        torch.tensor(
            -1e12,
            dtype=support.dtype,
            device=support.device,
        ),
        support,
    )
    best_idx = support.argmax()
    face = _vmap_select(faces, best_idx)
    normal = _vmap_select(normals, best_idx)

    edge_p0 = torch.roll(face, 1, dims=0)
    edge_p1 = face
    edge_normals = torch.vmap(math.cross, (0, None))(edge_p1 - edge_p0, normal)
    cap_pts_clipped, mask = _clip_edge_to_planes(cap_pts[0], cap_pts[1], edge_p0, edge_normals)
    cap_pts_clipped = cap_pts_clipped - normal * cap.geom_size[0]
    face_pts = torch.vmap(_project_pt_onto_plane, (0, None, None))(cap_pts_clipped, face[0], normal)
    pos = (cap_pts_clipped + face_pts) * 0.5
    norm = normal.unsqueeze(0).expand(2, -1)
    penetration = torch.where(
        mask,
        (face_pts - cap_pts_clipped) @ normal,
        torch.tensor(
            -1.0,
            dtype=face_pts.dtype,
            device=face_pts.device,
        ),
    )

    edge_closest, cap_closest = torch.vmap(math.closest_segment_to_segment_points, (0, 0, None, None))(
        edge_p0, edge_p1, cap_pts[0], cap_pts[1]
    )
    e_idx = ((edge_closest - cap_closest) ** 2).sum(dim=1).argmin()
    cap_closest_pt = _vmap_select(cap_closest, e_idx)
    edge_closest_pt = _vmap_select(edge_closest, e_idx)
    edge_axis = cap_closest_pt - edge_closest_pt
    edge_axis, edge_dist = math.normalize_with_norm(edge_axis)
    edge_pos = (edge_closest_pt + (cap_closest_pt - edge_axis * cap.geom_size[0])) * 0.5
    edge_penetration = cap.geom_size[0] - edge_dist
    has_edge_contact = edge_penetration > 0

    pos_updated = pos.clone()
    pos_updated[0] = edge_pos
    pos = torch.where(has_edge_contact, pos_updated, pos)
    norm_updated = norm.clone()
    norm_updated[0] = edge_axis
    n = -torch.where(has_edge_contact, norm_updated, norm)

    pos = prism.pos + pos @ prism.mat.T
    n = n @ prism.mat.T

    penetration_updated = penetration.clone()
    penetration_updated[0] = edge_penetration
    dist = -torch.where(has_edge_contact, penetration_updated, penetration)
    return dist, pos, n


def _convex_prism(obj: GeomInfo, prism: ConvexInfo) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convex vs prism. Returns (dist(4,), pos(4,3), n(4,3))."""
    obj_faces = obj.vert[obj.face]
    prism_faces = prism.face

    s1, s2 = obj_faces.shape[1], prism_faces.shape[1]
    if s1 < s2:
        obj_faces = F.pad(obj_faces, (0, 0, 0, s2 - s1), mode="replicate")
    elif s2 < s1:
        prism_faces = F.pad(prism_faces, (0, 0, 0, s1 - s2), mode="replicate")

    swapped = obj.vert.shape[0] > prism.vert.shape[0]
    if swapped:
        faces1, faces2 = prism_faces, obj_faces
        normals1, normals2 = prism.face_normal, obj.facenorm
        vert1, vert2 = prism.vert, obj.vert
        edge1, edge2 = prism.edge, obj.edge
        pos1, pos2 = prism.pos, obj.pos
        mat1, mat2 = prism.mat, obj.mat
    else:
        faces1, faces2 = obj_faces, prism_faces
        normals1, normals2 = obj.facenorm, prism.face_normal
        vert1, vert2 = obj.vert, prism.vert
        edge1, edge2 = obj.edge, prism.edge
        pos1, pos2 = obj.pos, prism.pos
        mat1, mat2 = obj.mat, prism.mat

    to_local_pos = mat2.T @ (pos1 - pos2)
    to_local_mat = mat2.T @ mat1

    local_faces1 = to_local_pos + faces1 @ to_local_mat.T
    local_normals1 = normals1 @ to_local_mat.T
    local_normals2 = normals2
    local_verts1 = to_local_pos + vert1 @ to_local_mat.T
    local_verts2 = vert2
    local_edges1 = local_verts1[edge1]
    local_edges2 = local_verts2[edge2]

    dist, pos, normal = _sat_hull_hull(
        local_faces1,
        faces2 if swapped else prism_faces,
        local_verts1,
        local_verts2,
        local_normals1,
        local_normals2,
        local_edges1,
        local_edges2,
    )

    pos = pos2 + pos @ mat2.T
    normal = normal @ mat2.T
    if swapped:
        normal = -normal
    return dist, pos, normal


def _build_prisms(
    h: HFieldInfo,
    obj_pos_local: torch.Tensor,
    obj_rbound: float,
    subgrid_size: tuple[int, int],
) -> list[ConvexInfo]:
    """Builds triangular prisms from the hfield sub-grid around the object."""
    dtype = obj_pos_local.dtype
    device = obj_pos_local.device

    xhalf = float(h.hfield_size[0])
    yhalf = float(h.hfield_size[1])
    zscale = float(h.hfield_size[2])
    zbase = float(h.hfield_size[3])

    dx = 2.0 * xhalf / (h.ncol - 1)
    dy = 2.0 * yhalf / (h.nrow - 1)

    xmin = float(obj_pos_local[0]) - obj_rbound
    ymin = float(obj_pos_local[1]) - obj_rbound
    xfrac = (xmin + xhalf) / (2 * xhalf) * (h.ncol - 1)
    yfrac = (ymin + yhalf) / (2 * yhalf) * (h.nrow - 1)
    cmin = int(np.floor(xfrac))
    rmin = int(np.floor(yfrac))

    prisms: list[ConvexInfo] = []
    for r in range(subgrid_size[1]):
        for c in range(subgrid_size[0]):
            ri = max(0, min(rmin + r, h.nrow - 2))
            ci = max(0, min(cmin + c, h.ncol - 2))

            def _pt(col, row):
                z = float(h.hfield_data[col, row])
                return torch.tensor(
                    [
                        dx * col - xhalf,
                        dy * row - yhalf,
                        z * zscale,
                    ],
                    dtype=dtype,
                    device=device,
                )

            p1 = _pt(ci, ri)
            p2 = _pt(ci + 1, ri + 1)
            p3 = _pt(ci, ri + 1)

            top = torch.stack([p1, p2, p3])
            bottom = top.clone()
            bottom[:, 2] = -zbase
            bottom = bottom[[0, 2, 1]]
            vert = torch.cat([bottom, top])
            prisms.append(mesh_module.hfield_prism(vert))

            p3b = p2
            p2b = _pt(ci + 1, ri)
            top = torch.stack([p1, p2b, p3b])
            bottom = top.clone()
            bottom[:, 2] = -zbase
            bottom = bottom[[0, 2, 1]]
            vert = torch.cat([bottom, top])
            prisms.append(mesh_module.hfield_prism(vert))

    return prisms


def _hfield_collision(
    collider_fn,
    h: HFieldInfo,
    obj: GeomInfo,
    obj_rbound: float,
    subgrid_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collides an object with prisms in a height field.

    Returns (dist, pos, n) in hfield local frame.
    """
    obj_pos = h.mat.T @ (obj.pos - h.pos)
    obj_mat = h.mat.T @ obj.mat
    obj_local = obj.replace(pos=obj_pos, mat=obj_mat)

    prisms = _build_prisms(h, obj_pos, obj_rbound, subgrid_size)

    all_dist: list[torch.Tensor] = []
    all_pos: list[torch.Tensor] = []
    all_n: list[torch.Tensor] = []
    for prism in prisms:
        d, p, n = collider_fn(obj_local, prism)
        all_dist.append(d.reshape(-1))
        all_pos.append(p.reshape(-1, 3))
        all_n.append(n.reshape(-1, 3))

    dist = torch.cat(all_dist)
    pos = torch.cat(all_pos)
    n = torch.cat(all_n)

    n = -n

    ncon_per_prism = all_dist[0].shape[0]
    top_norms = []
    for prism in prisms:
        tn = prism.face_normal[1]
        top_norms.extend([tn] * ncon_per_prism)
    top_norms = torch.stack(top_norms)

    cond = n[:, 2] < 1e-6
    n = torch.where(cond.unsqueeze(-1), top_norms, n)

    return dist, pos, n


def _select_manifold(dist, pos, n):
    """Selects 4 manifold contact points."""
    n_mean = torch.mean(n, dim=0)
    mask = dist < torch.minimum(torch.zeros_like(dist), dist.min() + _MANIFOLD_TOL.get(dist.dtype, dist.device))
    idx = _manifold_points(pos, mask, n_mean)
    dist = _vmap_take_1d(dist, idx)
    pos = _vmap_take(pos, idx)
    n = _vmap_take(n, idx)

    unique = torch.tril(idx == idx[:, None]).sum(dim=1) == 1
    dist = torch.where(unique, dist, torch.ones_like(dist))
    return dist, pos, n


def hfield_sphere(h: HFieldInfo, s: GeomInfo, subgrid_size: tuple[int, int]) -> Collision:
    """Calculates contacts between a hfield and a sphere."""
    rbound = float(torch.max(s.geom_size))
    dist, pos, n = _hfield_collision(_sphere_prism, h, s, rbound, subgrid_size)
    dist, pos, n = _select_manifold(dist, pos, n)

    pos = torch.vmap(lambda p: h.mat @ p + h.pos)(pos)
    n = torch.vmap(lambda nn: h.mat @ nn)(n)
    frame = torch.vmap(math.make_frame)(n)
    return dist, pos, frame


def hfield_capsule(h: HFieldInfo, c: GeomInfo, subgrid_size: tuple[int, int]) -> Collision:
    """Calculates contacts between a hfield and a capsule."""
    rbound = float(c.geom_size[0] + c.geom_size[1])
    dist, pos, n = _hfield_collision(_capsule_prism, h, c, rbound, subgrid_size)
    dist, pos, n = _select_manifold(dist, pos, n)

    pos = torch.vmap(lambda p: h.mat @ p + h.pos)(pos)
    n = torch.vmap(lambda nn: h.mat @ nn)(n)
    frame = torch.vmap(math.make_frame)(n)
    return dist, pos, frame


def hfield_convex(h: HFieldInfo, c: GeomInfo, subgrid_size: tuple[int, int]) -> Collision:
    """Calculates contacts between a hfield and a convex object (box/mesh)."""
    rbound = float(torch.max(c.geom_size))
    dist, pos, n = _hfield_collision(_convex_prism, h, c, rbound, subgrid_size)
    dist, pos, n = _select_manifold(dist, pos, n)

    pos = torch.vmap(lambda p: h.mat @ p + h.pos)(pos)
    n = torch.vmap(lambda nn: h.mat @ nn)(n)
    frame = torch.vmap(math.make_frame)(n)
    return dist, pos, frame


hfield_sphere.ncon = 4
hfield_capsule.ncon = 4
hfield_convex.ncon = 4
