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
"""Convex collisions."""

import torch
import torch.nn.functional as F

from mujoco_torch._src import math

# pylint: disable=g-importing-member
from mujoco_torch._src.collision_types import Collision as Contact
from mujoco_torch._src.collision_types import GeomInfo

# pylint: enable=g-importing-member


def _vmap_select(tensor: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Selects a single element along dim 0 using a scalar index, vmap-compatible."""
    if tensor.ndim == 1:
        return tensor.gather(0, idx.unsqueeze(0)).squeeze(0)
    idx_expanded = idx.view(1, *([1] * (tensor.ndim - 1))).expand(1, *tensor.shape[1:])
    return tensor.gather(0, idx_expanded).squeeze(0)


def _closest_segment_point_plane(
    a: torch.Tensor, b: torch.Tensor, p0: torch.Tensor, plane_normal: torch.Tensor
) -> torch.Tensor:
    """Gets the closest point between a line segment and a plane.

    Args:
      a: first line segment point
      b: second line segment point
      p0: point on plane
      plane_normal: plane normal

    Returns:
      closest point between the line segment and the plane
    """
    # Parametrize a line segment as S(t) = a + t * (b - a), plug it into the plane
    # equation dot(n, S(t)) - d = 0, then solve for t to get the line-plane
    # intersection. We then clip t to be in [0, 1] to be on the line segment.
    n = plane_normal
    d = torch.sum(p0 * n)  # shortest distance from origin to plane
    denom = torch.sum(n * (b - a))
    t = (d - torch.sum(n * a)) / (denom + 1e-6 * (denom == 0.0))
    t = torch.clamp(t, 0, 1)
    segment_point = a + t * (b - a)

    return segment_point


def _closest_triangle_point(p0: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Gets the closest point between a triangle and a point in space.

    Args:
      p0: triangle point
      p1: triangle point
      p2: triangle point
      pt: point to test

    Returns:
      closest point on the triangle w.r.t point pt
    """
    # Parametrize the triangle s.t. a point inside the triangle is
    # Q = p0 + u * e0 + v * e1, when 0 <= u <= 1, 0 <= v <= 1, and
    # 0 <= u + v <= 1. Let e0 = (p1 - p0) and e1 = (p2 - p0).
    # We analytically minimize the distance between the point pt and Q.
    e0 = p1 - p0
    e1 = p2 - p0
    a = torch.dot(e0, e0)
    b = torch.dot(e0, e1)
    c = torch.dot(e1, e1)
    d = pt - p0
    # The determinant is 0 only if the angle between e1 and e0 is 0
    # (i.e. the triangle has overlapping lines).
    det = a * c - b * b
    u = (c * torch.dot(e0, d) - b * torch.dot(e1, d)) / det
    v = (-b * torch.dot(e0, d) + a * torch.dot(e1, d)) / det
    inside = (0 <= u) & (u <= 1) & (0 <= v) & (v <= 1) & (u + v <= 1)
    closest_p = p0 + u * e0 + v * e1
    d0 = torch.dot(closest_p - pt, closest_p - pt)

    # If the closest point is outside the triangle, it must be on an edge, so we
    # check each triangle edge for a closest point to the point pt.
    closest_p1, d1 = math.closest_segment_point_and_dist(p0, p1, pt)
    closest_p = torch.where((d0 < d1) & inside, closest_p, closest_p1)
    min_d = torch.where((d0 < d1) & inside, d0, d1)

    closest_p2, d2 = math.closest_segment_point_and_dist(p1, p2, pt)
    closest_p = torch.where(d2 < min_d, closest_p2, closest_p)
    min_d = torch.minimum(min_d, d2)

    closest_p3, d3 = math.closest_segment_point_and_dist(p2, p0, pt)
    closest_p = torch.where(d3 < min_d, closest_p3, closest_p)

    return closest_p


def _closest_segment_triangle_points(
    a: torch.Tensor,
    b: torch.Tensor,
    p0: torch.Tensor,
    p1: torch.Tensor,
    p2: torch.Tensor,
    triangle_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gets the closest points between a line segment and triangle.

    Args:
      a: first line segment point
      b: second line segment point
      p0: triangle point
      p1: triangle point
      p2: triangle point
      triangle_normal: normal of triangle

    Returns:
      closest point on the triangle w.r.t the line segment
    """
    # The closest triangle point is either on the edge or within the triangle.
    # First check triangle edges for the closest point.
    # TODO(robotics-simulation): consider vmapping over closest point functions
    seg_pt1, tri_pt1 = math.closest_segment_to_segment_points(a, b, p0, p1)
    d1 = torch.dot(seg_pt1 - tri_pt1, seg_pt1 - tri_pt1)
    seg_pt2, tri_pt2 = math.closest_segment_to_segment_points(a, b, p1, p2)
    d2 = torch.dot(seg_pt2 - tri_pt2, seg_pt2 - tri_pt2)
    seg_pt3, tri_pt3 = math.closest_segment_to_segment_points(a, b, p0, p2)
    d3 = torch.dot(seg_pt3 - tri_pt3, seg_pt3 - tri_pt3)

    # Next, handle the case where the closest triangle point is inside the
    # triangle. Either the line segment intersects the triangle or a segment
    # endpoint is closest to a point inside the triangle.
    seg_pt4 = _closest_segment_point_plane(a, b, p0, triangle_normal)
    tri_pt4 = _closest_triangle_point(p0, p1, p2, seg_pt4)
    d4 = torch.dot(seg_pt4 - tri_pt4, seg_pt4 - tri_pt4)

    # Get the point with minimum distance from the line segment point to the
    # triangle point.
    distance = torch.stack([d1, d2, d3, d4]).unsqueeze(0)
    min_dist = torch.amin(distance)
    mask = (distance == min_dist).T
    seg_pt = torch.stack([seg_pt1, seg_pt2, seg_pt3, seg_pt4]) * mask
    tri_pt = torch.stack([tri_pt1, tri_pt2, tri_pt3, tri_pt4]) * mask
    seg_pt = torch.sum(seg_pt, dim=0) / torch.sum(mask)
    tri_pt = torch.sum(tri_pt, dim=0) / torch.sum(mask)

    return seg_pt, tri_pt


def _gather_idx(poly: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gathers a single row from poly using a scalar index, vmap-compatible."""
    return _vmap_select(poly, idx)


def _manifold_points(poly: torch.Tensor, poly_mask: torch.Tensor, poly_norm: torch.Tensor) -> torch.Tensor:
    """Chooses four points on the polygon with approximately maximal area."""
    dist_mask = torch.where(
        poly_mask,
        torch.zeros_like(poly_mask, dtype=poly.dtype),
        torch.full_like(poly_mask, -1e6, dtype=poly.dtype),
    )
    a_idx = torch.argmax(dist_mask)
    a = _gather_idx(poly, a_idx)
    # choose point b furthest from a
    b_idx = (((a - poly) ** 2).sum(dim=1) + dist_mask).argmax()
    b = _gather_idx(poly, b_idx)
    # choose point c furthest along the axis orthogonal to (a-b)
    ab = math.cross(poly_norm, a - b)
    ap = a - poly
    c_idx = (torch.abs(ap @ ab) + dist_mask).argmax()
    c = _gather_idx(poly, c_idx)
    # choose point d furthest from the other two triangle edges
    ac = math.cross(poly_norm, a - c)
    bc = math.cross(poly_norm, b - c)
    bp = b - poly
    dist_bp = torch.abs(bp @ bc) + dist_mask
    dist_ap = torch.abs(ap @ ac) + dist_mask
    d_idx = torch.cat([dist_bp, dist_ap]).argmax() % poly.shape[0]
    return torch.stack([a_idx, b_idx, c_idx, d_idx])


def _project_pt_onto_plane(pt: torch.Tensor, plane_pt: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Projects a point onto a plane along the plane normal."""
    dist = torch.dot(pt - plane_pt, plane_normal)
    return pt - dist * plane_normal


def _project_poly_onto_plane(poly: torch.Tensor, plane_pt: torch.Tensor, plane_normal: torch.Tensor) -> torch.Tensor:
    """Projects a polygon onto a plane using the plane normal."""
    return torch.vmap(_project_pt_onto_plane, (0, None, None))(poly, plane_pt, math.normalize(plane_normal))


def _project_poly_onto_poly_plane(
    poly1: torch.Tensor, norm1: torch.Tensor, poly2: torch.Tensor, norm2: torch.Tensor
) -> torch.Tensor:
    """Projects poly1 onto the poly2 plane along poly1's normal."""
    d = torch.dot(poly2[0], norm2)
    denom = torch.dot(norm1, norm2)
    t = (d - poly1 @ norm2) / (denom + 1e-6 * (denom == 0.0))
    new_poly = poly1 + t.reshape(-1, 1) * norm1
    return new_poly


def _point_in_front_of_plane(plane_pt: torch.Tensor, plane_normal: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Checks if a point is strictly in front of a plane."""
    return (pt - plane_pt) @ plane_normal > 1e-6


def _clip_edge_to_planes(
    edge_p0: torch.Tensor,
    edge_p1: torch.Tensor,
    plane_pts: torch.Tensor,
    plane_normals: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clips an edge against side planes.

    We return two clipped points, and a mask to include the new edge or not.

    Args:
      edge_p0: the first point on the edge
      edge_p1: the second point on the edge
      plane_pts: side plane points
      plane_normals: side plane normals

    Returns:
      new_ps: new edge points that are clipped against side planes
      mask: a boolean mask, True if an edge point is a valid clipped point and
      False otherwise
    """
    p0, p1 = edge_p0, edge_p1
    p0_in_front = torch.vmap(torch.dot)(p0 - plane_pts, plane_normals) > 1e-6
    p1_in_front = torch.vmap(torch.dot)(p1 - plane_pts, plane_normals) > 1e-6

    # Get candidate clipped points along line segment (p0, p1) by clipping against
    # all clipping planes.
    candidate_clipped_ps = torch.vmap(_closest_segment_point_plane, (None, None, 0, 0))(
        p0, p1, plane_pts, plane_normals
    )

    def clip_edge_point(p0, p1, p0_in_front, clipped_ps):
        @torch.vmap
        def choose_edge_point(in_front, clipped_p):
            return torch.where(in_front, clipped_p, p0)

        # Pick the clipped point if p0 is in front of the clipping plane. Otherwise
        # keep p0 as the edge point.
        new_edge_ps = choose_edge_point(p0_in_front, clipped_ps)

        # Pick the clipped point that is most along the edge direction.
        # This degenerates to picking the original point p0 if p0 is *not* in front
        # of any clipping planes.
        dists = (new_edge_ps - p0) @ (p1 - p0)
        new_edge_p = _vmap_select(new_edge_ps, torch.argmax(dists))
        return new_edge_p

    # Clip each edge point.
    new_p0 = clip_edge_point(p0, p1, p0_in_front, candidate_clipped_ps)
    new_p1 = clip_edge_point(p1, p0, p1_in_front, candidate_clipped_ps)
    clipped_pts = torch.stack([new_p0, new_p1])

    # Keep the original points if both points are in front of any of the clipping
    # planes, rather than creating a new clipped edge. If the entire subject edge
    # is in front of any clipping plane, we need to grab an edge from the clipping
    # polygon instead.
    both_in_front = p0_in_front & p1_in_front
    mask = ~torch.any(both_in_front)
    new_ps = torch.where(mask, clipped_pts, torch.stack([p0, p1]))
    # Mask out crossing clipped edge points.
    mask = torch.where(torch.dot(p0 - p1, new_ps[0] - new_ps[1]) < 0, False, mask)
    return new_ps, torch.stack([mask, mask])


def _clip(
    clipping_poly: torch.Tensor,
    subject_poly: torch.Tensor,
    clipping_normal: torch.Tensor,
    subject_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clips a subject polygon against a clipping polygon.

    A parallelized clipping algorithm for convex polygons. The result is a set of
    vertices on the clipped subject polygon in the subject polygon plane.

    Args:
      clipping_poly: the polygon that we use to clip the subject polygon against
      subject_poly: the polygon that gets clipped
      clipping_normal: normal of the clipping polygon
      subject_normal: normal of the subject polygon

    Returns:
      clipped_pts: points on the clipped polygon
      mask: True if a point is in the clipping polygon, False otherwise
    """
    # Get clipping edge points, edge planes, and edge normals.
    clipping_p0 = torch.roll(clipping_poly, 1, dims=0)
    clipping_plane_pts = clipping_p0
    clipping_p1 = clipping_poly
    clipping_plane_normals = torch.vmap(math.cross, (0, None))(
        clipping_p1 - clipping_p0,
        clipping_normal,
    )

    # Get subject edge points, edge planes, and edge normals.
    subject_edge_p0 = torch.roll(subject_poly, 1, dims=0)
    subject_plane_pts = subject_edge_p0
    subject_edge_p1 = subject_poly
    subject_plane_normals = torch.vmap(math.cross, (0, None))(
        subject_edge_p1 - subject_edge_p0,
        subject_normal,
    )

    # Clip all edges of the subject poly against clipping side planes.
    clipped_edges0, masks0 = torch.vmap(_clip_edge_to_planes, (0, 0, None, None))(
        subject_edge_p0,
        subject_edge_p1,
        clipping_plane_pts,
        clipping_plane_normals,
    )

    # Project the clipping poly onto the subject plane.
    clipping_p0_s = _project_poly_onto_poly_plane(clipping_p0, clipping_normal, subject_poly, subject_normal)
    clipping_p1_s = _project_poly_onto_poly_plane(clipping_p1, clipping_normal, subject_poly, subject_normal)

    # Clip all edges of the clipping poly against subject planes.
    clipped_edges1, masks1 = torch.vmap(_clip_edge_to_planes, (0, 0, None, None))(
        clipping_p0_s, clipping_p1_s, subject_plane_pts, subject_plane_normals
    )

    # Merge the points and reshape.
    clipped_edges = torch.cat([clipped_edges0, clipped_edges1])
    masks = torch.cat([masks0, masks1])
    clipped_points = clipped_edges.reshape((-1, 3))
    mask = masks.reshape(-1)

    return clipped_points, mask


def _create_contact_manifold(
    clipping_poly: torch.Tensor,
    subject_poly: torch.Tensor,
    clipping_norm: torch.Tensor,
    subject_norm: torch.Tensor,
    sep_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates a contact manifold between two convex polygons.

    The polygon faces are expected to have a counter clockwise winding order so
    that clipping plane normals point away from the polygon center.

    Args:
      clipping_poly: the reference polygon to clip the contact against.
      subject_poly: the subject polygon to clip contacts onto.
      clipping_norm: the clipping polygon normal.
      subject_norm: the subject polygon normal.
      sep_axis: the separating axis

    Returns:
      tuple of dist, pos, and normal
    """
    # Clip the subject (incident) face onto the clipping (reference) face.
    # The incident points are clipped points on the subject polygon.
    poly_incident, mask = _clip(clipping_poly, subject_poly, clipping_norm, subject_norm)
    # The reference points are clipped points on the clipping polygon.
    poly_ref = _project_poly_onto_plane(poly_incident, clipping_poly[0], clipping_norm)
    behind_clipping_plane = _point_in_front_of_plane(clipping_poly[0], -clipping_norm, poly_incident)
    mask = mask & behind_clipping_plane

    # Choose four contact points.
    best = _manifold_points(poly_ref, mask, clipping_norm)
    contact_pts = _vmap_take(poly_ref, best)
    mask_pts = _vmap_take_1d(mask, best)
    penetration_dir = _vmap_take(poly_incident, best) - contact_pts
    penetration = penetration_dir @ (-clipping_norm)

    dist = torch.where(mask_pts, -penetration, torch.ones_like(penetration))
    pos = contact_pts
    normal = -sep_axis.unsqueeze(0).expand(4, -1)
    return dist, pos, normal


def _vmap_take(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """vmap-compatible take along dim 0 for 2D tensor with 1D indices."""
    # indices: (K,), tensor: (N, D) -> result: (K, D)
    idx = indices.unsqueeze(-1).expand(-1, tensor.shape[-1])
    return torch.gather(tensor, 0, idx)


def _vmap_take_1d(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """vmap-compatible take along dim 0 for 1D tensor with 1D indices."""
    return torch.gather(tensor, 0, indices)


def _sat_hull_hull(
    faces_a: torch.Tensor,
    faces_b: torch.Tensor,
    vertices_a: torch.Tensor,
    vertices_b: torch.Tensor,
    normals_a: torch.Tensor,
    normals_b: torch.Tensor,
    unique_edges_a: torch.Tensor,
    unique_edges_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Runs the Separating Axis Test for a pair of hulls.

    Given two convex hulls, the Separating Axis Test finds a separating axis
    between all edge pairs and face pairs. Edge pairs create a single contact
    point and face pairs create a contact manifold (up to four contact points).
    We return both the edge and face contacts. Valid contacts can be checked with
    dist < 0. Resulting edge contacts should be preferred over face contacts.

    Args:
      faces_a: An ndarray of hull A's polygon faces.
      faces_b: An ndarray of hull B's polygon faces.
      vertices_a: Vertices for hull A.
      vertices_b: Vertices for hull B.
      normals_a: Normal vectors for hull A's polygon faces.
      normals_b: Normal vectors for hull B's polygon faces.
      unique_edges_a: Unique edges for hull A.
      unique_edges_b: Unique edges for hull B.

    Returns:
      tuple of dist, pos, and normal
    """
    # get the separating axes
    edge_dir_a = unique_edges_a[:, 0] - unique_edges_a[:, 1]
    edge_dir_b = unique_edges_b[:, 0] - unique_edges_b[:, 1]
    edge_dir_a_r = torch.tile(edge_dir_a, dims=(unique_edges_b.shape[0], 1))
    edge_dir_b_r = edge_dir_b.repeat_interleave(unique_edges_a.shape[0], dim=0)
    edge_edge_axes = torch.vmap(math.cross)(edge_dir_a_r, edge_dir_b_r)
    edge_edge_axes = torch.vmap(lambda x: math.normalize(x))(edge_edge_axes)

    axes = torch.cat([normals_a, normals_b, edge_edge_axes])

    # for each separating axis, get the support
    @torch.vmap
    def get_support(axis):
        support_a = torch.vmap(torch.dot, (None, 0))(axis, vertices_a)
        support_b = torch.vmap(torch.dot, (None, 0))(axis, vertices_b)
        dist1 = support_a.max() - support_b.min()
        dist2 = support_b.max() - support_a.min()
        sign = torch.where(dist1 > dist2, -1, 1)
        dist = torch.minimum(dist1, dist2)
        dist = torch.where(
            ~torch.all(axis == 0.0), dist, torch.tensor(1e6, dtype=dist.dtype, device=dist.device)
        )  # degenerate axis
        return dist, sign

    support, sign = get_support(axes)

    # choose the best separating axis
    best_idx = torch.argmin(support)
    best_sign = _vmap_select(sign, best_idx)
    best_axis = _vmap_select(axes, best_idx)
    is_edge_contact = best_idx >= (normals_a.shape[0] + normals_b.shape[0])

    # get the (reference) face most aligned with the separating axis
    dist_a = torch.vmap(torch.dot, (None, 0))(best_axis, normals_a)
    dist_b = torch.vmap(torch.dot, (None, 0))(best_axis, normals_b)
    a_max = dist_a.argmax()
    b_max = dist_b.argmax()
    a_min = dist_a.argmin()
    b_min = dist_b.argmin()

    ref_face = torch.where(best_sign > 0, _vmap_select(faces_a, a_max), _vmap_select(faces_b, b_max))
    ref_face_norm = torch.where(best_sign > 0, _vmap_select(normals_a, a_max), _vmap_select(normals_b, b_max))
    incident_face = torch.where(best_sign > 0, _vmap_select(faces_b, b_min), _vmap_select(faces_a, a_min))
    incident_face_norm = torch.where(best_sign > 0, _vmap_select(normals_b, b_min), _vmap_select(normals_a, a_min))

    dist, pos, normal = _create_contact_manifold(
        ref_face,
        incident_face,
        ref_face_norm,
        incident_face_norm,
        -best_sign * best_axis,
    )

    # For edge contacts, we use the clipped face point, mainly for performance
    # reasons. For small penetration, the clipped face point is roughly the edge
    # contact point.
    idx = dist.argmin()
    dist_at_idx = _vmap_select(dist, idx)
    pos_at_idx = _vmap_select(pos, idx)
    dist = torch.where(
        is_edge_contact,
        torch.stack(
            [dist_at_idx, torch.ones_like(dist_at_idx), torch.ones_like(dist_at_idx), torch.ones_like(dist_at_idx)]
        ),
        dist,
    )
    pos = torch.where(is_edge_contact, pos_at_idx.unsqueeze(0).expand(4, -1), pos)

    return dist, pos, normal


def plane_convex(plane: GeomInfo, convex: GeomInfo) -> Contact:
    """Calculates contacts between a plane and a convex object."""
    vert = convex.vert

    # get points in the convex frame
    plane_pos = convex.mat.T @ (plane.pos - convex.pos)
    n = convex.mat.T @ plane.mat[:, 2]
    support = (plane_pos - vert) @ n
    idx = _manifold_points(vert, support > 0, n)
    pos = _vmap_take(vert, idx)

    # convert to world frame
    pos = convex.pos + pos @ convex.mat.T
    n = plane.mat[:, 2]

    frame = torch.vmap(math.make_frame)(n.unsqueeze(0).expand(4, -1))
    unique = torch.tril(idx == idx[:, None]).sum(dim=1) == 1
    support_at_idx = _vmap_take_1d(support, idx)
    dist = torch.where(unique, -support_at_idx, torch.ones_like(support_at_idx))
    return dist, pos, frame


def sphere_convex(sphere: GeomInfo, convex: GeomInfo) -> Contact:
    """Calculates contact between a sphere and a convex object."""
    faces = convex.vert[convex.face]
    normals = convex.facenorm

    # Put sphere in convex frame.
    sphere_pos = convex.mat.T @ (sphere.pos - convex.pos)

    # Get support from face normals.
    @torch.vmap
    def get_support(faces, normal):
        pos = sphere_pos - normal * sphere.geom_size[0]
        return torch.dot(pos - faces[0], normal)

    support = get_support(faces, normals)

    # Pick the face with minimal penetration as long as it has support.
    support = torch.where(support >= 0, torch.tensor(-1e12, dtype=support.dtype, device=support.device), support)
    best_idx = support.argmax()
    face = _vmap_select(faces, best_idx)
    normal = _vmap_select(normals, best_idx)

    # Get closest point between the polygon face and the sphere center point.
    # Project the sphere center point onto poly plane. If it's inside polygon
    # edge normals, then we're done.
    pt = _project_pt_onto_plane(sphere_pos, face[0], normal)
    edge_p0 = torch.roll(face, 1, dims=0)
    edge_p1 = face
    edge_normals = torch.vmap(math.cross, (0, None))(
        edge_p1 - edge_p0,
        normal,
    )
    edge_dist = torch.vmap(lambda plane_pt, plane_norm: torch.dot(pt - plane_pt, plane_norm))(edge_p0, edge_normals)
    inside = torch.all(edge_dist <= 0)  # lte to handle degenerate edges

    # If the point is outside edge normals, project onto the closest edge plane
    # that the point is in front of.
    degenerate_edge = torch.all(edge_normals == 0, dim=1)
    behind = edge_dist < 0.0
    edge_dist = torch.where(
        degenerate_edge | behind, torch.tensor(1e12, dtype=edge_dist.dtype, device=edge_dist.device), edge_dist
    )
    idx = edge_dist.argmin()
    edge_pt = math.closest_segment_point(_vmap_select(edge_p0, idx), _vmap_select(edge_p1, idx), pt)

    pt = torch.where(inside, pt, edge_pt)

    # Get the normal, dist, and contact position.
    n, d = math.normalize_with_norm(pt - sphere_pos)
    spt = sphere_pos + n * sphere.geom_size[0]
    dist = d - sphere.geom_size[0]
    pos = (pt + spt) * 0.5

    # Go back to world frame.
    n = convex.mat @ n
    pos = convex.mat @ pos + convex.pos

    return torch.utils._pytree.tree_map(lambda x: torch.unsqueeze(x, 0), (dist, pos, math.make_frame(n)))


def capsule_convex(cap: GeomInfo, convex: GeomInfo) -> Contact:
    """Calculates contacts between a capsule and a convex object."""
    # Get convex transformed normals, faces, and vertices.
    faces = convex.vert[convex.face]
    normals = convex.facenorm

    # Put capsule in convex frame.
    cap_pos = convex.mat.T @ (cap.pos - convex.pos)
    axis, length = cap.mat[:, 2], cap.geom_size[1]
    axis = convex.mat.T @ axis
    seg = axis * length
    cap_pts = torch.stack(
        [
            cap_pos - seg,
            cap_pos + seg,
        ]
    )

    # Get support from face normals.
    @torch.vmap
    def get_support(face, normal):
        pts = cap_pts - normal * cap.geom_size[0]
        sup = torch.vmap(lambda x: torch.dot(x - face[0], normal))(pts)
        return sup.min()

    support = get_support(faces, normals)
    has_support = torch.all(support < 0)

    # Pick the face with minimal penetration as long as it has support.
    support = torch.where(support >= 0, torch.tensor(-1e12, dtype=support.dtype, device=support.device), support)
    best_idx = support.argmax()
    face = _vmap_select(faces, best_idx)
    normal = _vmap_select(normals, best_idx)

    # Clip the edge against side planes and create two contact points against the
    # face.
    edge_p0 = torch.roll(face, 1, dims=0)
    edge_p1 = face
    edge_normals = torch.vmap(math.cross, (0, None))(
        edge_p1 - edge_p0,
        normal,
    )
    cap_pts_clipped, mask = _clip_edge_to_planes(cap_pts[0], cap_pts[1], edge_p0, edge_normals)
    cap_pts_clipped = cap_pts_clipped - normal * cap.geom_size[0]
    face_pts = torch.vmap(_project_pt_onto_plane, (0, None, None))(cap_pts_clipped, face[0], normal)
    # Create variables for the face contact.
    pos = (cap_pts_clipped + face_pts) * 0.5
    norm = normal.unsqueeze(0).expand(2, -1)
    penetration = torch.where(
        mask & has_support,
        (face_pts - cap_pts_clipped) @ normal,
        torch.tensor(-1.0, dtype=face_pts.dtype, device=face_pts.device),
    )

    # Get a potential edge contact.
    edge_closest, cap_closest = torch.vmap(math.closest_segment_to_segment_points, (0, 0, None, None))(
        edge_p0, edge_p1, cap_pts[0], cap_pts[1]
    )
    e_idx = ((edge_closest - cap_closest) ** 2).sum(dim=1).argmin()
    cap_closest_pt = _vmap_select(cap_closest, e_idx)
    edge_closest_pt = _vmap_select(edge_closest, e_idx)
    edge_axis = cap_closest_pt - edge_closest_pt
    edge_axis, edge_dist = math.normalize_with_norm(edge_axis)
    edge_pos = (edge_closest_pt + (cap_closest_pt - edge_axis * cap.geom_size[0])) * 0.5
    edge_norm = edge_axis
    edge_penetration = cap.geom_size[0] - edge_dist
    has_edge_contact = edge_penetration > 0

    # Get the contact info.
    pos_updated = pos.clone()
    pos_updated[0] = edge_pos
    pos = torch.where(has_edge_contact, pos_updated, pos)
    norm_updated = norm.clone()
    norm_updated[0] = edge_norm
    n = -torch.where(has_edge_contact, norm_updated, norm)

    # Go back to world frame.
    pos = convex.pos + pos @ convex.mat.T
    n = n @ convex.mat.T

    penetration_updated = penetration.clone()
    penetration_updated[0] = edge_penetration
    dist = -torch.where(has_edge_contact, penetration_updated, penetration)
    frame = torch.vmap(math.make_frame)(n)
    return dist, pos, frame


def convex_convex(c1: GeomInfo, c2: GeomInfo) -> Contact:
    """Calculates contacts between two convex objects."""
    if c1.face is None or c2.face is None or c1.vert is None or c2.vert is None:
        raise AssertionError("Mesh info missing.")
    # pad face vertices so that we can broadcast between geom1 and geom2
    s1, s2 = c1.face.shape[-1], c2.face.shape[-1]
    if s1 < s2:
        face = F.pad(c1.face, (0, s2 - s1), mode="replicate")
        c1 = c1.replace(face=face)
    elif s2 < s1:
        face = F.pad(c2.face, (0, s1 - s2), mode="replicate")
        c2 = c2.replace(face=face)

    # ensure that the first object has fewer verts
    swapped = c1.vert.shape[0] > c2.vert.shape[0]
    if swapped:
        c1, c2 = c2, c1

    faces1 = c1.vert[c1.face]
    faces2 = c2.vert[c2.face]

    to_local_pos = c2.mat.T @ (c1.pos - c2.pos)
    to_local_mat = c2.mat.T @ c1.mat

    faces1 = to_local_pos + faces1 @ to_local_mat.T
    normals1 = c1.facenorm @ to_local_mat.T
    normals2 = c2.facenorm

    vertices1 = to_local_pos + c1.vert @ to_local_mat.T
    vertices2 = c2.vert

    unique_edges1 = vertices1[c1.edge]
    unique_edges2 = vertices2[c2.edge]

    dist, pos, normal = _sat_hull_hull(
        faces1,
        faces2,
        vertices1,
        vertices2,
        normals1,
        normals2,
        unique_edges1,
        unique_edges2,
    )

    # Go back to world frame.
    pos = c2.pos + pos @ c2.mat.T
    normal = normal @ c2.mat.T
    normal = -normal if swapped else normal

    frame = torch.vmap(math.make_frame)(normal)
    return dist, pos, frame


# store ncon as function attributes
plane_convex.ncon = 4
sphere_convex.ncon = 1
capsule_convex.ncon = 2
convex_convex.ncon = 4
