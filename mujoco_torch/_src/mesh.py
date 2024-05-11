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
import dataclasses
import itertools
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import mujoco
import torch
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import GeomType
from mujoco.mjx._src.types import Model
# pylint: enable=g-importing-member
import numpy as np
from scipy import spatial
import trimesh
from tensordict import tensorclass

from mujoco_torch._src.math import concatenate
from mujoco_torch._src.types import pytree_repr, NonVmappableTensor

_BOX_CORNERS = list(itertools.product((-1, 1), (-1, 1), (-1, 1)))
# pyformat: disable
# Rectangular box faces using a counter-clockwise winding order convention.
_BOX_FACES = [
    0, 4, 5, 1,  # left
    0, 2, 6, 4,  # bottom
    6, 7, 5, 4,  # front
    2, 3, 7, 6,  # right
    1, 5, 7, 3,  # top
    0, 1, 3, 2,  # back
]
# pyformat: enable
_MAX_HULL_FACE_VERTICES = 20
_CONVEX_CACHE: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
_DERIVED_ARGS = [
    'geom_convex_face',
    'geom_convex_vert',
    'geom_convex_edge_dir',
    'geom_convex_facenormal',
    'geom_convex_edge',
    'geom_convex_edge_face_normal',
]
DERIVED = {(Model, d) for d in _DERIVED_ARGS}


def _box(size: torch.Tensor):
  """Creates a mesh for a box with rectangular faces."""
  box_corners = torch.tensor(_BOX_CORNERS)
  vert = box_corners * size.reshape(-1, 3)
  face = torch.tensor([_BOX_FACES]).reshape(-1, 4)
  return vert, face


def _get_face_norm(vert: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
  """Calculates face normals given vertices and face indexes."""
  assert len(vert.shape) == 2 and len(face.shape) == 2, (
      f'vert and face should have dim of 2, got {len(vert.shape)} and '
      f'{len(face.shape)}'
  )
  face_vert = vert[face, :]
  # use CCW winding order convention
  edge0 = face_vert[:, 1, :] - face_vert[:, 0, :]
  edge1 = face_vert[:, -1, :] - face_vert[:, 0, :]
  face_norm = torch.cross(edge0, edge1)
  face_norm = face_norm / torch.linalg.norm(face_norm, dim=1).reshape((-1, 1))
  return face_norm


def _get_unique_edge_dir(vert: torch.Tensor, face: torch.Tensor) -> torch.Tensor:
  """Returns unique edge directions.

  Args:
    vert: (n_vert, 3) vertices
    face: (n_face, n_vert) face index array

  Returns:
    edges: tuples of vertex indexes for each edge
  """
  r_face = torch.roll(face, shifts=1, dims=1)
  edges = concatenate(torch.stack([face, r_face], 1))

  # do a first pass to remove duplicates
  edges.sort(dim=1)
  edges = torch.unique(edges, dim=0)
  edges = edges[edges[:, 0] != edges[:, 1]]  # get rid of edges from padded face

  # get normalized edge directions
  edge_vert = vert[edges]
  edge_dir = edge_vert[:, 0] - edge_vert[:, 1]
  norms = torch.sqrt(torch.sum(edge_dir**2, dim=1))
  edge_dir = edge_dir / norms.reshape((-1, 1))

  # get the first unique edge for all pairwise comparisons
  diff1 = edge_dir[:, None, :] - edge_dir[None, :, :]
  diff2 = edge_dir[:, None, :] + edge_dir[None, :, :]
  matches = (torch.linalg.norm(diff1, dim=-1) < 1e-6) | (
      torch.linalg.norm(diff2, dim=-1) < 1e-6
  )
  matches = torch.tril(matches).sum(dim=-1)
  unique_edge_idx = torch.where(matches == 1)[0]

  return edges[unique_edge_idx]


def _get_edge_normals(
    face: torch.Tensor, face_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns face edges and face edge normals."""
  # get face edges and scatter the face norms
  r_face = torch.roll(face, shifts=1, dims=1)
  face_edge = torch.stack([face, r_face], -1)
  face_edge.sort(dim=2)
  face_edge_flat = concatenate(face_edge)
  edge_face_idx = torch.repeat_interleave(torch.arange(face.shape[0]), face.shape[1])
  edge_face_norm = face_norm[edge_face_idx]

  # get the edge normals associated with each edge
  edge_map_list = collections.defaultdict(list)
  for i in range(face_edge_flat.shape[0]):
    if face_edge_flat[i][0] == face_edge_flat[i][1]:
      continue
    edge_map_list[tuple(face_edge_flat[i])].append(edge_face_norm[i])

  edges, edge_face_normals = [], []
  for k, v in edge_map_list.items():
    v = torch.stack(v)
    if len(v) > 2:
      # Meshes can be of poor quality and contain edges adjacent to more than
      # two faces. We take the first two unique face normals.
      v = torch.unique(v, dim=0)[:2]
    elif len(v) == 1:
      # Some edges are either degenerate or _MAX_HULL_FACE_VERTICES was hit
      # and face vertices were down sampled. In either case, we ignore these
      # edges.
      continue
    edges.append(k)
    edge_face_normals.append(v)

  return torch.tensor(edges), torch.tensor(edge_face_normals)

def _convex_hull_2d(points: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
  """Calculates the convex hull for a set of points on a plane."""
  # project points onto the closest axis plane
  best_axis = torch.abs(torch.vmap(torch.dot, (0, None))(torch.eye(3, dtype=normal.dtype), normal)).argmax()
  axis = torch.eye(3)[best_axis]
  d = torch.vmap(torch.dot, (0, None))(points, axis.to(points.dtype)).reshape((-1, 1))
  # d = torch.vmap(torch.dot, (0, None))(points, axis).reshape((-1, 1))
  axis_points = points - d * axis
  axis_points = axis_points[:, list({0, 1, 2} - {best_axis.tolist()})]

  # get the polygon face, and make the points ccw wrt the face normal
  c = spatial.ConvexHull(axis_points)
  order_ = torch.where(axis.to(normal.dtype).dot(normal) > 0, 1, -1)
  order_ *= torch.where(best_axis == 1, -1, 1)
  hull_point_idx = torch.as_tensor(c.vertices[::order_].copy())
  assert (axis_points - c.points).sum() == 0

  return hull_point_idx


@tensorclass(autocast=True)
class MeshInfo:
  name: str
  vert: torch.Tensor
  face: torch.Tensor
  convex_vert: torch.Tensor = None
  convex_face: torch.Tensor = None


def _merge_coplanar(tm: trimesh.Trimesh, mesh_info: MeshInfo) -> torch.Tensor:
  """Merges coplanar facets."""
  if not tm.facets:
    return tm.faces.copy()  # no facets
  if not tm.faces.shape[0]:
    raise ValueError('Mesh has no faces.')

  # Get faces.
  face_idx = set(range(tm.faces.shape[0])) - set(np.concatenate(tm.facets))
  face_idx = torch.tensor(list(face_idx))
  faces = tm.faces[face_idx] if face_idx.shape[0] > 0 else torch.tensor([])

  # Get facets.
  facets = []
  tm_faces = torch.tensor(np.asarray(tm.faces))
  for i, facet in enumerate(tm.facets):
    face = tm_faces[facet]
    point_idx = torch.unique(face)
    points = torch.tensor(tm.vertices[point_idx])
    normal = torch.tensor(tm.facets_normal[i])

    # convert triangulated facet to a polygon
    hull_point_idx = _convex_hull_2d(points, normal)
    face = point_idx[hull_point_idx.contiguous()]

    # resize faces that exceed max polygon vertices
    if face.shape[0] > _MAX_HULL_FACE_VERTICES:
      warnings.warn(
          f'Mesh "{mesh_info.name}" has a coplanar face with more than'
          f' {_MAX_HULL_FACE_VERTICES} vertices. This may lead to performance '
          'issues and inaccuracies in collision detection. Consider '
          'decimating the mesh.'
      )
    every = face.shape[0] // _MAX_HULL_FACE_VERTICES + 1
    face = face[::every]
    facets.append(face)

  # Pad facets so that they can be stacked.
  max_len = max(f.shape[0] for f in facets) if facets else faces.shape[1]
  assert max_len <= _MAX_HULL_FACE_VERTICES
  for i, f in enumerate(facets):
    if f.shape[0] < max_len:
      f = torch.pad(f, (0, max_len - f.shape[0]), 'edge')
    facets[i] = f

  if not faces.shape[0]:
    assert len(facets)
    return torch.stack(facets)  # no faces, return facets

  # Merge faces and facets.
  faces = torch.pad(faces, ((0, 0), (0, max_len - faces.shape[1])), 'edge')
  return concatenate([faces, facets])


def _mesh_info(
    m: mujoco.MjModel,
) -> List[MeshInfo]:
  """Extracts mesh info from MjModel."""
  mesh_infos = []
  for i in range(m.nmesh):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH.value, i)

    last = (i + 1) >= m.nmesh
    face_start = m.mesh_faceadr[i]
    face_end = m.mesh_faceadr[i + 1] if not last else m.mesh_face.shape[0]
    face = m.mesh_face[face_start:face_end]

    vert_start = m.mesh_vertadr[i]
    vert_end = m.mesh_vertadr[i + 1] if not last else m.mesh_vert.shape[0]
    vert = m.mesh_vert[vert_start:vert_end]

    graphadr = m.mesh_graphadr[i]
    if graphadr < 0:
      mesh_infos.append(MeshInfo(name, vert, face, None, None))
      continue

    graph = m.mesh_graph[graphadr:]
    numvert, numface = graph[0], graph[1]

    # unused vert_edgeadr
    # vert_edgeadr = graph[2 : numvert + 2]
    last_idx = numvert + 2

    vert_globalid = graph[last_idx : last_idx + numvert]
    last_idx += numvert

    # unused edge_localid
    # edge_localid = graph[last_idx : last_idx + numvert + 3 * numface]
    last_idx += numvert + 3 * numface

    face_globalid = graph[last_idx : last_idx + 3 * numface]
    face_globalid = face_globalid.reshape((numface, 3))

    convex_vert = vert[vert_globalid]
    vertex_map = dict(zip(vert_globalid, torch.arange(vert_globalid.shape[0])))
    convex_face = torch.vectorize(vertex_map.get)(face_globalid)
    mesh_infos.append(MeshInfo(name, vert, face, convex_vert, convex_face))

  return mesh_infos


def _geom_mesh_kwargs(
    mesh_info: MeshInfo,
) -> Dict[str, torch.Tensor]:
  """Generates convex mesh attributes for mjx.Model."""
  tm_convex = trimesh.Trimesh(
      vertices=mesh_info.convex_vert, faces=mesh_info.convex_face
  )
  vert = torch.tensor(tm_convex.vertices)
  face = _merge_coplanar(tm_convex, mesh_info)
  facenormal = _get_face_norm(vert, face)
  edge, edge_face_normal = _get_edge_normals(face, facenormal)
  return {
      'geom_convex_face': vert[face],
      'geom_convex_face_vert_idx': face,
      'geom_convex_vert': vert,
      'geom_convex_edge_dir': _get_unique_edge_dir(vert, face),
      'geom_convex_facenormal': facenormal,
      'geom_convex_edge': edge,
      'geom_convex_edge_face_normal': edge_face_normal,
  }


def get(m: mujoco.MjModel) -> Dict[str, Sequence[Optional[torch.Tensor]]]:
  """Derives geom mesh attributes for mjx.Model from MjModel."""
  kwargs = {k: [] for k in _DERIVED_ARGS}
  mesh_infos = _mesh_info(m)
  geom_con = m.geom_conaffinity | m.geom_contype
  for geomid in range(m.ngeom):
    mesh_info = None
    dataid = m.geom_dataid[geomid]
    if not geom_con[geomid]:
      # ignore visual-only meshes
      kwargs = {k: kwargs[k] + [None] for k in _DERIVED_ARGS}
      continue
    elif m.geom_type[geomid] == GeomType.BOX:
      vert, face = _box(m.geom_size[geomid])
      mesh_info = MeshInfo(
          name='box',
          vert=vert,
          face=face,
          convex_vert=vert,
          convex_face=face,
      )
    elif dataid < 0:
      kwargs = {k: kwargs[k] + [None] for k in _DERIVED_ARGS}
      continue

    mesh_info = mesh_info if mesh_info is not None else mesh_infos[dataid]
    vert, face = mesh_info.vert, mesh_info.face
    # TODO: use other hash method?
    key = (hash(vert.data.numpy().tobytes()), hash(face.data.numpy().tobytes()))
    if key not in _CONVEX_CACHE:
      _CONVEX_CACHE[key] = _geom_mesh_kwargs(mesh_info)

    kwargs = {k: kwargs[k] + [_CONVEX_CACHE[key][k]] for k in _DERIVED_ARGS}

  for key in list(kwargs):
    try:
      kwargs[key] = NonVmappableTensor(torch.stack(kwargs[key]))
    except (RuntimeError, TypeError):
      shape, dtype = next(iter((tensor.shape, tensor.dtype) for tensor in kwargs[key] if isinstance(tensor, torch.Tensor)))
      shape = torch.Size([0, *shape[1:]])
      vals = kwargs[key]
      vals = [val if val is not None else torch.zeros(shape, dtype=dtype) for val in vals]
      kwargs[key] = NonVmappableTensor(torch.nested.nested_tensor(vals))
  return kwargs
