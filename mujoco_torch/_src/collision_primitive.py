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
"""Collision primitives."""

import torch

from mujoco_torch._src import math

# pylint: disable=g-importing-member
from mujoco_torch._src.collision_types import Collision as Contact
from mujoco_torch._src.collision_types import GeomInfo
from mujoco_torch._src.diff_config import get_diff_config

# pylint: enable=g-importing-member


def _plane_sphere(
    plane_normal: torch.Tensor,
    plane_pos: torch.Tensor,
    sphere_pos: torch.Tensor,
    sphere_radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the distance and contact point between a plane and sphere."""
    dist = ((sphere_pos - plane_pos) * plane_normal).sum(-1) - sphere_radius
    pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
    return dist, pos


def plane_sphere(plane: GeomInfo, sphere: GeomInfo) -> Contact:
    """Calculates contact between a plane and a sphere."""
    n = plane.mat[:, 2]
    dist, pos = _plane_sphere(n, plane.pos, sphere.pos, sphere.geom_size[0])
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def plane_capsule(plane: GeomInfo, cap: GeomInfo) -> Contact:
    """Calculates two contacts between a capsule and a plane."""
    cfg = get_diff_config()
    n, axis = plane.mat[:, 2], cap.mat[:, 2]
    # align contact frames with capsule axis
    b, b_norm = math.normalize_with_norm(axis - n * (n * axis).sum(-1))
    y, z = (
        torch.eye(3, dtype=axis.dtype, device=axis.device)[1],
        torch.eye(3, dtype=axis.dtype, device=axis.device)[2],
    )
    if cfg.smooth_collisions:
        s = cfg.smooth_sharpness
        w_valid = torch.sigmoid(s * (b_norm - 0.5))
        w_yz = torch.sigmoid(s * (torch.abs(n[1]) - 0.5))
        fallback = math.soft_where(w_yz, z, y)
        b = math.soft_where(w_valid, b, fallback)
    else:
        b = torch.where(b_norm < 0.5, torch.where((-0.5 < n[1]) & (n[1] < 0.5), y, z), b)
    frame = torch.stack([n, b, math.cross(n, b)]).unsqueeze(0)
    segment = axis * cap.geom_size[1]
    contacts = []
    for offset in [segment, -segment]:
        dist, pos = _plane_sphere(n, plane.pos, cap.pos + offset, cap.geom_size[0])
        dist = torch.unsqueeze(dist, 0)
        pos = torch.unsqueeze(pos, 0)
        contacts.append((dist, pos, frame))
    return torch.utils._pytree.tree_map(lambda *x: torch.cat(x, dim=0), *contacts)


def plane_ellipsoid(plane: GeomInfo, ellipsoid: GeomInfo) -> Contact:
    """Calculates one contact between an ellipsoid and a plane."""
    n = plane.mat[:, 2]
    size = ellipsoid.geom_size
    sphere_support = -math.normalize((ellipsoid.mat.T * n).sum(-1) * size)
    pos = ellipsoid.pos + (ellipsoid.mat * (sphere_support * size)).sum(-1)
    dist = (n * (pos - plane.pos)).sum(-1)
    pos = pos - n * dist * 0.5
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def plane_cylinder(plane: GeomInfo, cylinder: GeomInfo) -> Contact:
    """Calculates three contacts between a cylinder and a plane."""
    cfg = get_diff_config()
    n = plane.mat[:, 2]
    axis = cylinder.mat[:, 2]

    # make sure axis points towards plane
    prjaxis = (n * axis).sum(-1)
    sign = -math.soft_sign(prjaxis, cfg.smooth_sharpness) if cfg.smooth_collisions else -math.sign(prjaxis)
    axis = axis * sign
    prjaxis = prjaxis * sign

    # compute normal distance to cylinder center
    dist0 = ((cylinder.pos - plane.pos) * n).sum(-1)

    # remove component of -normal along axis, compute length
    vec = axis * prjaxis - n
    len_ = math.norm(vec)

    if cfg.smooth_collisions:
        s = cfg.smooth_sharpness
        _zero = torch.zeros((), dtype=len_.dtype, device=len_.device)
        _eps = torch.full((), 1e-12, dtype=len_.dtype, device=len_.device)
        w_len = torch.sigmoid(s * (len_ - _eps))
        vec = math.soft_where(
            w_len,
            math.safe_div(vec, len_) * cylinder.geom_size[0],
            cylinder.mat[:, 0] * cylinder.geom_size[0],
        )
    else:
        vec = torch.where(
            len_ < 1e-12,
            cylinder.mat[:, 0] * cylinder.geom_size[0],
            math.safe_div(vec, len_) * cylinder.geom_size[0],
        )

    # project vector on normal
    prjvec = (vec * n).sum(-1)

    # scale axis by half-length
    axis = axis * cylinder.geom_size[1]
    prjaxis = prjaxis * cylinder.geom_size[1]

    # compute sideways vector: vec1
    prjvec1 = -prjvec * 0.5
    vec1 = math.normalize(math.cross(vec, axis)) * cylinder.geom_size[0]
    vec1 = vec1 * (torch.sqrt(torch.full((), 3.0, device=vec.device)) * 0.5)

    # disk parallel to plane
    d1 = dist0 + prjaxis + prjvec
    d2 = dist0 + prjaxis + prjvec1
    dist_disk = torch.stack([d1, d2, d2])
    pos_disk = (
        cylinder.pos
        + axis
        + torch.stack(
            [
                vec - n * d1 * 0.5,
                vec1 + vec * -0.5 - n * d2 * 0.5,
                -vec1 + vec * -0.5 - n * d2 * 0.5,
            ]
        )
    )

    # cylinder parallel to plane
    d3 = dist0 - prjaxis + prjvec
    dist_par = torch.stack([d1, d3, d2])
    pos_par = torch.stack([pos_disk[0], cylinder.pos + vec - axis - n * d3 * 0.5, pos_disk[2]])

    if cfg.smooth_collisions:
        s = cfg.smooth_sharpness
        _thresh = torch.full((), 1e-3, dtype=prjaxis.dtype, device=prjaxis.device)
        w_par = torch.sigmoid(s * (_thresh - torch.abs(prjaxis)))
        dist = math.soft_where(w_par.unsqueeze(0), dist_par, dist_disk)
        pos = math.soft_where(w_par.unsqueeze(0).unsqueeze(-1), pos_par, pos_disk)
    else:
        cond = torch.abs(prjaxis) < 1e-3
        dist = torch.where(cond, dist_par, dist_disk)
        pos = torch.where(cond, pos_par, pos_disk)

    frame = torch.stack([math.make_frame(n)] * 3, dim=0)
    return dist, pos, frame


def _sphere_sphere(
    pos1: torch.Tensor, radius1: torch.Tensor, pos2: torch.Tensor, radius2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the penetration, contact point, and normal between two spheres."""
    cfg = get_diff_config()
    n, dist = math.normalize_with_norm(pos2 - pos1)
    if cfg.smooth_collisions:
        _eps = torch.full((), 1e-6, dtype=dist.dtype, device=dist.device)
        w = torch.sigmoid(cfg.smooth_sharpness * (dist - _eps))
        n = math.soft_where(w, n, torch.eye(3, dtype=n.dtype, device=n.device)[0])
    else:
        n = torch.where(dist == 0.0, torch.eye(3, dtype=n.dtype, device=n.device)[0], n)
    dist = dist - (radius1 + radius2)
    pos = pos1 + n * (radius1 + dist * 0.5)
    return dist, pos, n


def sphere_sphere(s1: GeomInfo, s2: GeomInfo) -> Contact:
    """Calculates contact between two spheres."""
    dist, pos, n = _sphere_sphere(s1.pos, s1.geom_size[0], s2.pos, s2.geom_size[0])
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def sphere_capsule(sphere: GeomInfo, cap: GeomInfo) -> Contact:
    """Calculates one contact between a sphere and a capsule."""
    axis, length = cap.mat[:, 2], cap.geom_size[1]
    segment = axis * length
    pt = math.closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)
    dist, pos, n = _sphere_sphere(sphere.pos, sphere.geom_size[0], pt, cap.geom_size[0])
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def capsule_capsule(cap1: GeomInfo, cap2: GeomInfo) -> Contact:
    """Calculates one contact between two capsules."""
    axis1, length1, axis2, length2 = (
        cap1.mat[:, 2],
        cap1.geom_size[1],
        cap2.mat[:, 2],
        cap2.geom_size[1],
    )
    seg1, seg2 = axis1 * length1, axis2 * length2
    pt1, pt2 = math.closest_segment_to_segment_points(
        cap1.pos - seg1,
        cap1.pos + seg1,
        cap2.pos - seg2,
        cap2.pos + seg2,
    )
    radius1, radius2 = cap1.geom_size[0], cap2.geom_size[0]
    dist, pos, n = _sphere_sphere(pt1, radius1, pt2, radius2)
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


# store ncon as function attributes
plane_sphere.ncon = 1
plane_capsule.ncon = 2
plane_ellipsoid.ncon = 1
plane_cylinder.ncon = 3
sphere_sphere.ncon = 1
sphere_capsule.ncon = 1
capsule_capsule.ncon = 1
