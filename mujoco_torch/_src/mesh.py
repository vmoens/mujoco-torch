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
"""Mesh processing."""

import collections
import itertools
import warnings
from collections.abc import Sequence

import mujoco
import numpy as np
import torch
import trimesh
from scipy import spatial

from mujoco_torch._src import math
from mujoco_torch._src.collision_types import ConvexInfo, GeomInfo, HFieldInfo
from mujoco_torch._src.types import ConvexMesh, GeomType, Model

_MAX_HULL_FACE_VERTICES = 20
_DERIVED_ARGS = [
    "geom_convex_face",
    "geom_convex_vert",
    "geom_convex_edge",
    "geom_convex_facenormal",
]
_CONVEX_CACHE: dict[tuple[int, int], dict[str, np.ndarray]] = {}
DERIVED = {(Model, d) for d in _DERIVED_ARGS}


def _get_face_norm(vert: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Calculates face normals given vertices and face indexes."""
    assert len(vert.shape) == 2 and len(face.shape) == 2, (
        f"vert and face should have dim of 2, got {len(vert.shape)} and {len(face.shape)}"
    )
    face_vert = vert[face, :]
    # use CCW winding order convention
    edge0 = face_vert[:, 1, :] - face_vert[:, 0, :]
    edge1 = face_vert[:, -1, :] - face_vert[:, 0, :]
    face_norm = np.cross(edge0, edge1)
    face_norm = face_norm / np.linalg.norm(face_norm, axis=1).reshape((-1, 1))
    return face_norm


def _get_edge_normals(face: np.ndarray, face_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns face edges and face edge normals."""
    # get face edges and scatter the face norms
    r_face = np.roll(face, 1, axis=1)
    face_edge = np.array([face, r_face]).transpose((1, 2, 0))
    face_edge.sort(axis=2)
    face_edge_flat = np.concatenate(face_edge)
    edge_face_idx = np.repeat(np.arange(face.shape[0]), face.shape[1])
    edge_face_norm = face_norm[edge_face_idx]

    # get the edge normals associated with each edge
    edge_map_list = collections.defaultdict(list)
    for i in range(face_edge_flat.shape[0]):
        if face_edge_flat[i][0] == face_edge_flat[i][1]:
            continue
        edge_map_list[tuple(face_edge_flat[i])].append(edge_face_norm[i])

    edges, edge_face_normals = [], []
    for k, v in edge_map_list.items():
        v = np.array(v)
        if len(v) > 2:
            # Meshes can be of poor quality and contain edges adjacent to more than
            # two faces. We take the first two unique face normals.
            v = np.unique(v, axis=0)[:2]
        elif len(v) == 1:
            # Some edges are either degenerate or _MAX_HULL_FACE_VERTICES was hit
            # and face vertices were down sampled. In either case, we ignore these
            # edges.
            continue
        edges.append(k)
        edge_face_normals.append(v)

    return np.array(edges), np.array(edge_face_normals)


def _convex_hull_2d(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Calculates the convex hull for a set of points on a plane."""
    # project points onto the closest axis plane
    best_axis = np.abs(np.eye(3).dot(normal)).argmax()
    axis = np.eye(3)[best_axis]
    d = points.dot(axis).reshape((-1, 1))
    axis_points = points - d * axis
    axis_points = axis_points[:, list({0, 1, 2} - {best_axis})]

    # get the polygon face, and make the points ccw wrt the face normal
    c = spatial.ConvexHull(axis_points)
    order_ = np.where(axis.dot(normal) > 0, 1, -1)
    order_ *= np.where(best_axis == 1, -1, 1)
    hull_point_idx = c.vertices[::order_]
    assert (axis_points - c.points).sum() == 0

    return hull_point_idx


def _merge_coplanar(m: mujoco.MjModel | Model, tm: trimesh.Trimesh, meshid: int) -> np.ndarray:
    """Merges coplanar facets."""
    if not tm.facets:
        return tm.faces.copy()  # no facets
    if not tm.faces.shape[0]:
        raise ValueError("Mesh has no faces.")

    # Get faces.
    face_idx = set(range(tm.faces.shape[0])) - set(np.concatenate(tm.facets))
    face_idx = np.array(list(face_idx))
    faces = tm.faces[face_idx] if face_idx.shape[0] > 0 else np.array([])

    # Get facets.
    facets = []
    for i, facet in enumerate(tm.facets):
        point_idx = np.unique(tm.faces[facet])
        points = tm.vertices[point_idx]
        normal = tm.facets_normal[i]

        # convert triangulated facet to a polygon
        hull_point_idx = _convex_hull_2d(points, normal)
        face = point_idx[hull_point_idx]

        # resize faces that exceed max polygon vertices
        if face.shape[0] > _MAX_HULL_FACE_VERTICES:
            name = m.names[m.name_meshadr[meshid] :]
            name = name[: name.find(b"\x00")].decode("utf-8")
            warnings.warn(
                f'Mesh "{name}" has a coplanar face with more than '
                f"{_MAX_HULL_FACE_VERTICES} vertices. This may lead to performance "
                "issues and inaccuracies in collision detection. Consider "
                "decimating the mesh."
            )
            every = face.shape[0] // _MAX_HULL_FACE_VERTICES + 1
            face = face[::every]
        facets.append(face)

    # Pad facets so that they can be stacked.
    max_len = max(f.shape[0] for f in facets) if facets else faces.shape[1]
    assert max_len <= _MAX_HULL_FACE_VERTICES
    for i, f in enumerate(facets):
        if f.shape[0] < max_len:
            f = np.pad(f, (0, max_len - f.shape[0]), "edge")
            facets[i] = f

    if not faces.shape[0]:
        assert facets
        return np.array(facets)  # no faces, return facets

    # Merge faces and facets.
    faces = np.pad(faces, ((0, 0), (0, max_len - faces.shape[1])), "edge")
    return np.concatenate([faces, facets])


def box(info: GeomInfo) -> ConvexInfo:
    """Creates a box with rectangular faces."""
    vert = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))), dtype=float)
    # pyformat: disable
    # rectangular box faces using a counter-clockwise winding order convention:
    face = np.array(
        [
            0,
            4,
            5,
            1,  # left
            0,
            2,
            6,
            4,  # bottom
            6,
            7,
            5,
            4,  # front
            2,
            3,
            7,
            6,  # right
            1,
            5,
            7,
            3,  # top
            0,
            1,
            3,
            2,  # back
        ]
    ).reshape((-1, 4))
    # pyformat: enable
    face_normal = _get_face_norm(vert, face)
    edge, edge_face_normal = _get_edge_normals(face, face_normal)
    face = vert[face]  # materialize full nface x nvert matrix

    c = ConvexInfo(
        info.pos,
        info.mat,
        geom_size=info.geom_size,
        vert=torch.tensor(vert, dtype=info.pos.dtype),
        face=torch.tensor(face, dtype=info.pos.dtype),
        face_normal=torch.tensor(face_normal, dtype=info.pos.dtype),
        edge=torch.tensor(edge, dtype=torch.int64),
        edge_face_normal=torch.tensor(edge_face_normal, dtype=info.pos.dtype),
        batch_size=[],
    )
    vert = c.vert * info.geom_size
    face = c.face * info.geom_size
    c = c.replace(vert=vert, face=face)

    return c


def convex(m: mujoco.MjModel | Model, data_id: int) -> ConvexMesh:
    """Processes a mesh for use in convex collision algorithms.

    Args:
      m: an MJX model
      data_id: the mesh id to process

    Returns:
      a convex mesh
    """
    vert_beg = m.mesh_vertadr[data_id]
    vert_end = m.mesh_vertadr[data_id + 1] if data_id < m.nmesh - 1 else None
    vert = m.mesh_vert[vert_beg:vert_end]

    graphadr = m.mesh_graphadr[data_id]
    graph = m.mesh_graph[graphadr:]
    graph_idx = 0

    numvert, numface = graph[0], graph[1]
    graph_idx += 2

    # skip vert_edgeadr (numvert,)
    graph_idx += numvert
    vert_globalid = graph[graph_idx : graph_idx + numvert]
    graph_idx += numvert

    # skip edge_localid (numvert, 3)
    graph_idx += numvert + 3 * numface
    face_globalid = graph[graph_idx : graph_idx + 3 * numface].reshape((-1, 3))

    vert = vert[vert_globalid]
    vertex_map = dict(zip(vert_globalid.tolist(), np.arange(vert_globalid.shape[0])))
    face = np.vectorize(vertex_map.get)(face_globalid)

    tm_convex = trimesh.Trimesh(vertices=vert, faces=face)
    vert = np.array(tm_convex.vertices)
    face = _merge_coplanar(m, tm_convex, data_id)
    face_normal = _get_face_norm(vert, face)
    edge, edge_face_normal = _get_edge_normals(face, face_normal)
    face = vert[face]  # materialize full nface x nvert matrix

    c = ConvexMesh(
        torch.tensor(vert, dtype=torch.float64),
        torch.tensor(face, dtype=torch.float64),
        torch.tensor(face_normal, dtype=torch.float64),
        torch.tensor(edge, dtype=torch.int64),
        torch.tensor(edge_face_normal, dtype=torch.float64),
        batch_size=[],
    )

    return c


def hfield_prism(vert: torch.Tensor) -> ConvexInfo:
    """Builds a hfield prism."""
    # The first 3 vertices define the bottom triangle, and the next 3 vertices
    # define the top triangle. The remaining triangles define the side of the
    # prism.
    face = np.array(
        [
            [0, 1, 2, 0],  # bottom
            [3, 4, 5, 3],  # top
            [0, 3, 5, 1],
            [0, 2, 4, 3],
            [2, 1, 5, 4],
        ]
    )
    edges = np.array(
        [
            # bottom
            [0, 1],
            [1, 2],
            [0, 2],
            # top
            [3, 4],
            [3, 5],
            [4, 5],
            # sides
            [0, 3],
            [1, 5],
            [2, 4],
        ]
    )
    edge_face_norm = np.array(
        [
            # bottom
            [0, 2],
            [0, 4],
            [0, 3],
            # top
            [1, 3],
            [1, 2],
            [1, 4],
            # sides
            [2, 3],
            [2, 4],
            [3, 4],
        ]
    )

    def get_face_norm(face):
        # use ccw winding order convention, and avoid using the last vertex
        edge0 = face[2, :] - face[1, :]
        edge1 = face[0, :] - face[1, :]
        return math.normalize(torch.linalg.cross(edge0, edge1))

    centroid = torch.mean(vert, dim=0)
    vert = vert - centroid
    face_t = vert[face]
    face_norm = torch.stack([get_face_norm(face_t[i]) for i in range(face_t.shape[0])])

    c = ConvexInfo(
        centroid,
        torch.eye(3, dtype=vert.dtype, device=vert.device),
        geom_size=torch.ones(3, dtype=vert.dtype, device=vert.device),
        vert=vert,
        face=face_t,
        face_normal=face_norm,
        edge=torch.tensor(edges, dtype=torch.int64, device=vert.device),
        edge_face_normal=face_norm[edge_face_norm],
        batch_size=[],
    )

    return c


def hfield(m: mujoco.MjModel | Model, data_id: int) -> HFieldInfo:
    adr = int(m.hfield_adr[data_id])
    nrow = int(m.hfield_nrow[data_id])
    ncol = int(m.hfield_ncol[data_id])
    raw = m.hfield_data[adr : adr + nrow * ncol]
    if isinstance(raw, torch.Tensor):
        raw = raw.numpy()
    data = np.array(raw).reshape(
        (ncol, nrow), order="F"
    )
    h = HFieldInfo(
        torch.zeros(3, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        hfield_size=np.asarray(m.hfield_size[data_id]),
        nrow=nrow,
        ncol=ncol,
        hfield_data=torch.tensor(data, dtype=torch.float64),
        batch_size=[],
    )
    return h


def _get_unique_edges(vert: np.ndarray, face: np.ndarray) -> np.ndarray:
    """Returns unique edges (vertex index pairs)."""
    edge, _ = _get_edge_normals(face, _get_face_norm(vert, face))
    return edge


def _get_faces_verts(
    m: mujoco.MjModel,
) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
    """Extracts mesh faces and vertices from MjModel."""
    verts, faces = [], []
    for i in range(m.nmesh):
        last = (i + 1) >= m.nmesh
        face_start = m.mesh_faceadr[i]
        face_end = m.mesh_faceadr[i + 1] if not last else m.mesh_face.shape[0]
        face = m.mesh_face[face_start:face_end]
        faces.append(face)

        vert_start = m.mesh_vertadr[i]
        vert_end = m.mesh_vertadr[i + 1] if not last else m.mesh_vert.shape[0]
        vert = m.mesh_vert[vert_start:vert_end]
        verts.append(vert)
    return verts, faces


def _geom_mesh_kwargs(m: mujoco.MjModel, vert: np.ndarray, face: np.ndarray, meshid: int) -> dict[str, np.ndarray]:
    """Generates convex mesh attributes for mjx.Model."""
    tm = trimesh.Trimesh(vertices=vert, faces=face)
    tm_convex = trimesh.convex.convex_hull(tm)
    vert = np.array(tm_convex.vertices)
    face = _merge_coplanar(m, tm_convex, max(0, meshid))
    return {
        "geom_convex_face": face,
        "geom_convex_vert": vert,
        "geom_convex_edge": _get_unique_edges(vert, face),
        "geom_convex_facenormal": _get_face_norm(vert, face),
    }


def get(m: mujoco.MjModel) -> dict[str, Sequence[np.ndarray | None]]:
    """Derives geom mesh attributes for mjx.Model from MjModel."""
    kwargs = {k: [] for k in _DERIVED_ARGS}
    verts, faces = _get_faces_verts(m)
    for geomid in range(m.ngeom):
        dataid = m.geom_dataid[geomid]
        typ = m.geom_type[geomid]
        if typ == GeomType.BOX:
            box_corners = np.array(list(itertools.product((-1, 1), (-1, 1), (-1, 1))))
            vert = box_corners * m.geom_size[geomid].reshape(-1, 3)
            face = np.array(
                [
                    [0, 4, 5, 1],
                    [0, 2, 6, 4],
                    [6, 7, 5, 4],
                    [2, 3, 7, 6],
                    [1, 5, 7, 3],
                    [0, 1, 3, 2],
                ]
            ).reshape(-1, 4)
            key = (hash(vert.data.tobytes()), hash(face.data.tobytes()))
            meshid = -1  # box has no mesh
        elif typ == GeomType.HFIELD:
            kwargs = {
                k: kwargs[k] + [None]
                for k in _DERIVED_ARGS
            }
            continue
        elif dataid >= 0:
            vert, face = verts[dataid], faces[dataid]
            key = (
                hash(vert.data.tobytes()),
                hash(face.data.tobytes()),
            )
            meshid = dataid
        else:
            kwargs = {k: kwargs[k] + [None] for k in _DERIVED_ARGS}
            continue

        if key not in _CONVEX_CACHE:
            _CONVEX_CACHE[key] = _geom_mesh_kwargs(m, vert, face, meshid)

        kwargs = {k: kwargs[k] + [_CONVEX_CACHE[key][k]] for k in _DERIVED_ARGS}

    return kwargs
