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

# pylint: enable=g-importing-member


def _plane_sphere(
    plane_normal: torch.Tensor,
    plane_pos: torch.Tensor,
    sphere_pos: torch.Tensor,
    sphere_radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the distance and contact point between a plane and sphere."""
    dist = torch.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
    pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
    return dist, pos


def plane_sphere(plane: GeomInfo, sphere: GeomInfo) -> Contact:
    """Calculates contact between a plane and a sphere."""
    n = plane.mat[:, 2]
    dist, pos = _plane_sphere(n, plane.pos, sphere.pos, sphere.geom_size[0])
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def plane_capsule(plane: GeomInfo, cap: GeomInfo) -> Contact:
    """Calculates two contacts between a capsule and a plane."""
    n, axis = plane.mat[:, 2], cap.mat[:, 2]
    # align contact frames with capsule axis
    b, b_norm = math.normalize_with_norm(axis - n * torch.dot(n, axis))
    y, z = (
        torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device),
        torch.tensor([0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device),
    )
    b = torch.where(b_norm < 0.5, torch.where((-0.5 < n[1]) & (n[1] < 0.5), y, z), b)
    frame = torch.stack([n, b, torch.linalg.cross(n, b)]).unsqueeze(0)
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
    sphere_support = -math.normalize((ellipsoid.mat.T @ n) * size)
    pos = ellipsoid.pos + ellipsoid.mat @ (sphere_support * size)
    dist = torch.dot(n, pos - plane.pos)
    pos = pos - n * dist * 0.5
    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def plane_cylinder(plane: GeomInfo, cylinder: GeomInfo) -> Contact:
    """Calculates three contacts between a cylinder and a plane."""
    n = plane.mat[:, 2]
    axis = cylinder.mat[:, 2]

    # make sure axis points towards plane
    prjaxis = torch.dot(n, axis)
    sign = -math.sign(prjaxis)
    axis = axis * sign
    prjaxis = prjaxis * sign

    # compute normal distance to cylinder center
    dist0 = torch.dot(cylinder.pos - plane.pos, n)

    # remove component of -normal along axis, compute length
    vec = axis * prjaxis - n
    len_ = math.norm(vec)

    vec = torch.where(
        len_ < 1e-12,
        cylinder.mat[:, 0] * cylinder.geom_size[0],
        math.safe_div(vec, len_) * cylinder.geom_size[0],
    )

    # project vector on normal
    prjvec = torch.dot(vec, n)

    # scale axis by half-length
    axis = axis * cylinder.geom_size[1]
    prjaxis = prjaxis * cylinder.geom_size[1]

    # compute sideways vector: vec1
    prjvec1 = -prjvec * 0.5
    vec1 = math.normalize(torch.linalg.cross(vec, axis)) * cylinder.geom_size[0]
    vec1 = vec1 * (torch.sqrt(torch.tensor(3.0, device=vec.device)) * 0.5)

    # disk parallel to plane
    d1 = dist0 + prjaxis + prjvec
    d2 = dist0 + prjaxis + prjvec1
    dist = torch.stack([d1, d2, d2])
    pos = (
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
    cond = torch.abs(prjaxis) < 1e-3
    d3 = dist0 - prjaxis + prjvec
    dist_new = dist.clone()
    dist_new = torch.stack([dist_new[0], d3, dist_new[2]])
    dist = torch.where(cond, dist_new, dist)
    pos_new = pos.clone()
    pos_new = torch.stack([pos_new[0], cylinder.pos + vec - axis - n * d3 * 0.5, pos_new[2]])
    pos = torch.where(cond, pos_new, pos)

    frame = torch.stack([math.make_frame(n)] * 3, dim=0)
    return dist, pos, frame


def _sphere_sphere(
    pos1: torch.Tensor, radius1: torch.Tensor, pos2: torch.Tensor, radius2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns the penetration, contact point, and normal between two spheres."""
    n, dist = math.normalize_with_norm(pos2 - pos1)
    n = torch.where(dist == 0.0, torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device), n)
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
