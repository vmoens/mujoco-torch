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
"""Collide geometries."""

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import mujoco
# pylint: enable=g-importing-member
import numpy as np
# from torch import numpy as torch
import torch
from mujoco_torch._src import collision_base
# pylint: disable=g-importing-member
from mujoco_torch._src.collision_base import Candidate
from mujoco_torch._src.collision_base import CandidateSet
from mujoco_torch._src.collision_base import GeomInfo
from mujoco_torch._src.collision_base import SolverParams
from mujoco_torch._src.collision_convex import capsule_convex
from mujoco_torch._src.collision_convex import convex_convex
from mujoco_torch._src.collision_convex import plane_convex
from mujoco_torch._src.collision_convex import sphere_convex
from mujoco_torch._src.collision_primitive import capsule_capsule
from mujoco_torch._src.collision_primitive import plane_capsule
from mujoco_torch._src.collision_primitive import plane_sphere
from mujoco_torch._src.collision_primitive import sphere_capsule
from mujoco_torch._src.collision_primitive import sphere_sphere
from mujoco_torch._src.types import Contact
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import GeomType
from mujoco_torch._src.types import Model

# pair-wise collision functions
_COLLISION_FUNC = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
    (GeomType.PLANE, GeomType.BOX): plane_convex,
    (GeomType.PLANE, GeomType.MESH): plane_convex,
    (GeomType.SPHERE, GeomType.SPHERE): sphere_sphere,
    (GeomType.SPHERE, GeomType.CAPSULE): sphere_capsule,
    (GeomType.SPHERE, GeomType.BOX): sphere_convex,
    (GeomType.SPHERE, GeomType.MESH): sphere_convex,
    (GeomType.CAPSULE, GeomType.CAPSULE): capsule_capsule,
    (GeomType.CAPSULE, GeomType.BOX): capsule_convex,
    (GeomType.CAPSULE, GeomType.MESH): capsule_convex,
    (GeomType.BOX, GeomType.BOX): convex_convex,
    (GeomType.BOX, GeomType.MESH): convex_convex,
    (GeomType.MESH, GeomType.MESH): convex_convex,
}


def get_collision_fn(
    key: Tuple[Union[GeomType, mujoco.mjtGeom], Union[GeomType, mujoco.mjtGeom]]
) -> Optional[Callable[[GeomInfo, GeomInfo], collision_base.Contact]]:
  """Returns a collision function given a pair of geom types."""
  return _COLLISION_FUNC.get(key, None)


def _add_candidate(
    result: CandidateSet,
    m: Union[Model, mujoco.MjModel],
    g1: int,
    g2: int,
    ipair: int = -1,
):
  """Adds a candidate to test for collision."""
  t1, t2 = m.geom_type[g1], m.geom_type[g2]
  if t1 > t2:
    t1, t2, g1, g2 = t2, t1, g2, g1

  def mesh_key(i):
    convex_data = [[None] * m.ngeom] * 3
    if isinstance(m, Model):
      convex_data = [m.geom_convex_face, m.geom_convex_vert, m.geom_convex_edge]
    key = tuple((-1,) if v[i] is None else v[i].shape for v in convex_data)
    return key

  k1, k2 = mesh_key(g1), mesh_key(g2)

  candidates = {(c.geom1, c.geom2) for c in result.get((t1, t2, k1, k2), [])}
  if (g1, g2) in candidates:
    return

  if ipair > -1:
    candidate = Candidate(g1, g2, ipair, -1, m.pair_dim[ipair])
  elif m.geom_priority[g1] != m.geom_priority[g2]:
    gp = g1 if m.geom_priority[g1] > m.geom_priority[g2] else g2
    candidate = Candidate(g1, g2, -1, gp, m.geom_condim[gp])
  else:
    dim = max(m.geom_condim[g1], m.geom_condim[g2])
    candidate = Candidate(g1, g2, -1, -1, dim)

  result.setdefault((t1, t2, k1, k2), []).append(candidate)


def _pair_params(
    m: Model,
    candidates: Sequence[Candidate],
) -> SolverParams:
  """Gets solver params for pair geoms."""
  ipair = torch.tensor([c.ipair for c in candidates])
  friction = torch.clamp_min(m.pair_friction[ipair], mujoco.mjMINMU)
  solref = m.pair_solref[ipair]
  solreffriction = m.pair_solreffriction[ipair]
  solimp = m.pair_solimp[ipair]
  margin = m.pair_margin[ipair]
  gap = m.pair_gap[ipair]

  return SolverParams(friction, solref, solreffriction, solimp, margin, gap)


def _priority_params(
    m: Model,
    candidates: Sequence[Candidate],
) -> SolverParams:
  """Gets solver params from priority geoms."""
  geomp = torch.tensor([c.geomp for c in candidates])
  friction = m.geom_friction[geomp][:, torch.tensor([0, 0, 1, 2, 2])]
  solref = m.geom_solref[geomp]
  solreffriction = torch.zeros(geomp.shape + (mujoco.mjNREF,))
  solimp = m.geom_solimp[geomp]
  g = torch.tensor([(c.geom1, c.geom2) for c in candidates])
  margin = torch.amax(m.geom_margin[g.T], axis=0)
  gap = torch.amax(m.geom_gap[g.T], axis=0)

  return SolverParams(friction, solref, solreffriction, solimp, margin, gap)


def _dynamic_params(
    m: Model,
    candidates: Sequence[Candidate],
) -> SolverParams:
  """Gets solver params for dynamic geoms."""
  g1 = torch.tensor([c.geom1 for c in candidates])
  g2 = torch.tensor([c.geom2 for c in candidates])

  friction = torch.maximum(m.geom_friction[g1], m.geom_friction[g2])
  # copy friction terms for the full geom pair
  friction = friction[:, torch.tensor([0, 0, 1, 2, 2])]

  minval = torch.tensor(mujoco.mjMINVAL)
  solmix1, solmix2 = m.geom_solmix[g1], m.geom_solmix[g2]
  mix = solmix1 / (solmix1 + solmix2)
  mix = torch.where((solmix1 < minval) & (solmix2 < minval), 0.5, mix)
  mix = torch.where((solmix1 < minval) & (solmix2 >= minval), 0.0, mix)
  mix_fn = torch.vmap(lambda a, b, m: m * a + (1 - m) * b)

  solref1, solref2 = m.geom_solref[g1], m.geom_solref[g2]
  solref = torch.minimum(solref1, solref2)
  s_mix = mix_fn(solref1, solref2, mix)
  solref = torch.where((solref1[0] > 0) & (solref2[0] > 0), s_mix, solref)
  solreffriction = torch.zeros(g1.shape + (mujoco.mjNREF,))
  solimp = mix_fn(m.geom_solimp[g1], m.geom_solimp[g2], mix)
  margin = torch.maximum(m.geom_margin[g1], m.geom_margin[g2])
  gap = torch.maximum(m.geom_gap[g1], m.geom_gap[g2])

  return SolverParams(friction, solref, solreffriction, solimp, margin, gap)


def _pair_info(
    m: Model, d: Data, geom1: Sequence[int], geom2: Sequence[int]
) -> Tuple[GeomInfo, GeomInfo, Sequence[Dict[str, Optional[int]]]]:
  """Returns geom pair info for calculating collision."""
  g1, g2 = torch.tensor(geom1), torch.tensor(geom2)
  info1 = GeomInfo(
      d.geom_xpos[g1],
      d.geom_xmat[g1],
      m.geom_size[g1],
  )
  info2 = GeomInfo(
      d.geom_xpos[g2],
      d.geom_xmat[g2],
      m.geom_size[g2],
  )
  in_axes1 = in_axes2 = torch.utils._pytree.tree_map(lambda x: 0, info1)
  if m.geom_convex_face[geom1[0]] is not None:
    info1 = info1.replace(
        face=torch.stack([m.geom_convex_face[i] for i in geom1]),
        vert=torch.stack([m.geom_convex_vert[i] for i in geom1]),
        edge=torch.stack([m.geom_convex_edge[i] for i in geom1]),
        facenorm=torch.stack([m.geom_convex_facenormal[i] for i in geom1]),
    )
    in_axes1 = in_axes1.replace(face=0, vert=0, edge=0, facenorm=0)
  if m.geom_convex_face[geom2[0]] is not None:
    info2 = info2.replace(
        face=torch.stack([m.geom_convex_face[i] for i in geom2]),
        vert=torch.stack([m.geom_convex_vert[i] for i in geom2]),
        edge=torch.stack([m.geom_convex_edge[i] for i in geom2]),
        facenorm=torch.stack([m.geom_convex_facenormal[i] for i in geom2]),
    )
    in_axes2 = in_axes2.replace(face=0, vert=0, edge=0, facenorm=0)
  return info1, info2, (in_axes1, in_axes2)


def _body_pair_filter(
    m: Union[Model, mujoco.MjModel], b1: int, b2: int
) -> bool:
  """Filters body pairs for collision."""
  dsbl_filterparent = m.opt.disableflags & DisableBit.FILTERPARENT
  weld1 = m.body_weldid[b1]
  weld2 = m.body_weldid[b2]
  parent_weld1 = m.body_weldid[m.body_parentid[weld1]]
  parent_weld2 = m.body_weldid[m.body_parentid[weld2]]

  if weld1 == weld2:
    # filter out self-collisions
    return True

  if (
      not dsbl_filterparent
      and weld1 != 0
      and weld2 != 0
      and (weld1 == parent_weld2 or weld2 == parent_weld1)
  ):
    # filter out parent-child collisions
    return True

  return False


def _collide_geoms(
    m: Model,
    d: Data,
    geom_types: Tuple[GeomType, GeomType],
    candidates: Sequence[Candidate],
) -> Contact:
  """Collides a geom pair."""
  fn = get_collision_fn(geom_types)
  if not fn:
    return Contact.zero()

  # group sol params by different candidate types
  typ_cands = {}
  for c in candidates:
    typ = (c.ipair > -1, c.geomp > -1)
    typ_cands.setdefault(typ, []).append(c)

  geom1, geom2, params = [], [], []
  for (pair, priority), candidates in typ_cands.items():
    geom1.extend([c.geom1 for c in candidates])
    geom2.extend([c.geom2 for c in candidates])
    if pair:
      params.append(_pair_params(m, candidates))
    elif priority:
      params.append(_priority_params(m, candidates))
    else:
      params.append(_dynamic_params(m, candidates))

  # call contact function
  g1, g2, in_axes = _pair_info(m, d, geom1, geom2)
  res = torch.vmap(fn, in_axes)(g1, g2)
  dist, pos, frame = torch.utils._pytree.tree_map(lambda *x: mujoco_torch._src.math.contatenate(x), *res)

  params = torch.utils._pytree.tree_map(lambda *x: mujoco_torch._src.math.contatenate(x), *params)
  geom1, geom2 = torch.tensor(geom1), torch.tensor(geom2)
  # repeat params by the number of contacts per geom pair
  n_repeat = dist.shape[-1] // geom1.shape[0]
  geom1, geom2, params = torch.utils._pytree.tree_map(
      lambda x: x.repeat_interleave(n_repeat, dim=0),
      (geom1, geom2, params),
  )

  con = Contact(
      dist=dist,
      pos=pos,
      frame=frame,
      includemargin=params.margin - params.gap,
      friction=params.friction,
      solref=params.solref,
      solreffriction=params.solreffriction,
      solimp=params.solimp,
      geom1=geom1,
      geom2=geom2,
      dim=np.array([]),
      efc_address=np.array([]),
  )
  return con


def _max_contact_points(m: Model) -> int:
  """Returns the maximum number of contact points when set as a numeric."""
  for i in range(m.nnumeric):
    name = m.names[m.name_numericadr[i] :].decode('utf-8').split('\x00', 1)[0]
    if name == 'max_contact_points':
      return int(m.numeric_data[m.numeric_adr[i]])

  return -1


def collision_candidates(m: Union[Model, mujoco.MjModel]) -> CandidateSet:
  """Returns candidates for collision checking."""
  candidate_set = {}

  for ipair in range(m.npair):
    g1, g2 = m.pair_geom1[ipair], m.pair_geom2[ipair]
    _add_candidate(candidate_set, m, g1, g2, ipair)

  body_pairs = []
  exclude_signature = set(m.exclude_signature)
  for b1 in range(m.nbody):
    for b2 in range(b1, m.nbody):
      signature = (b1 << 16) + (b2)
      if signature in exclude_signature:
        continue
      if _body_pair_filter(m, b1, b2):
        continue
      body_pairs.append((b1, b2))

  for b1, b2 in body_pairs:
    start1 = m.body_geomadr[b1]
    end1 = m.body_geomadr[b1] + m.body_geomnum[b1]
    for g1 in range(start1, end1):
      start2 = m.body_geomadr[b2]
      end2 = m.body_geomadr[b2] + m.body_geomnum[b2]
      for g2 in range(start2, end2):
        mask = m.geom_contype[g1] & m.geom_conaffinity[g2]
        mask |= m.geom_contype[g2] & m.geom_conaffinity[g1]
        if mask != 0:
          _add_candidate(candidate_set, m, g1, g2)

  return candidate_set


def ncon(m: Model) -> int:
  """Returns the number of contacts computed in MJX given a model."""
  if m.opt.disableflags & DisableBit.CONTACT:
    return 0

  candidates = collision_candidates(m)
  max_count = _max_contact_points(m)

  count = 0
  for k, v in candidates.items():
    fn = get_collision_fn(k[0:2])
    if fn is None:
      continue
    count += len(v) * fn.ncon  # pytype: disable=attribute-error

  return min(max_count, count) if max_count > -1 else count


def collision(m: Model, d: Data) -> Data:
  """Collides geometries."""
  ncon_ = ncon(m)
  if ncon_ == 0:
    return d.replace(contact=Contact.zero(), ncon=0)

  candidate_set = collision_candidates(m)

  contacts = []
  for key, candidates in candidate_set.items():
    geom_types = key[0:2]
    contacts.append(_collide_geoms(m, d, geom_types, candidates))

  if not contacts:
    raise RuntimeError('No contacts found.')

  contact = torch.utils._pytree.tree_map(lambda *x: mujoco_torch._src.math.contatenate(x), *contacts)

  max_contact_points = _max_contact_points(m)
  if max_contact_points > -1 and contact.dist.shape[0] > max_contact_points:
    # get top-k contacts
    _, idx = torch.lax.top_k(-contact.dist, k=max_contact_points)
    contact = torch.utils._pytree.tree_map(lambda x, idx=idx: torch.take(x, idx, axis=0), contact)

  if ncon_ != contact.dist.shape[0]:
    raise RuntimeError('Number of contacts does not match ncon.')

  # TODO(robotics-simulation): move this logic to device_put
  ns = d.ne + d.nf + d.nl
  contact = contact.replace(efc_address=np.arange(ns, ns + ncon_ * 4, 4))
  # TODO(robotics-simulation): add support for other friction dimensions
  contact = contact.replace(dim=3 * np.ones(ncon_, dtype=np.int32))

  return d.replace(contact=contact, ncon=ncon_)
