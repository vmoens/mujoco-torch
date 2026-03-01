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

import dataclasses
import math as _math
from collections.abc import Callable, Sequence

import mujoco

# pylint: enable=g-importing-member
import numpy as np

# from torch import numpy as torch
import torch

from mujoco_torch._src import collision_types
from mujoco_torch._src import mesh as mesh_module
from mujoco_torch._src.collision_convex import capsule_convex, convex_convex, plane_convex, sphere_convex
from mujoco_torch._src.collision_hfield import hfield_capsule, hfield_convex, hfield_sphere
from mujoco_torch._src.collision_primitive import (
    capsule_capsule,
    plane_capsule,
    plane_sphere,
    sphere_capsule,
    sphere_sphere,
)

# pylint: disable=g-importing-member
from mujoco_torch._src.collision_types import GeomInfo
from mujoco_torch._src.dataclasses import MjTensorClass
from mujoco_torch._src.math import _CachedConst
from mujoco_torch._src.scan import _DeviceCachedTensor
from mujoco_torch._src.types import Contact, Data, DisableBit, GeomType, Model

_FRICTION_IDX = _CachedConst([0, 0, 1, 2, 2])
_MINVAL = _CachedConst(mujoco.mjMINVAL)


@dataclasses.dataclass(frozen=True)
class Candidate:
    """A geom pair candidate for collision testing."""

    geom1: int
    geom2: int
    ipair: int
    geomp: int
    dim: int


class SolverParams(MjTensorClass):
    """Solver parameters for a collision contact."""

    friction: torch.Tensor
    solref: torch.Tensor
    solreffriction: torch.Tensor
    solimp: torch.Tensor
    margin: torch.Tensor
    gap: torch.Tensor


CandidateSet = dict[tuple, list[Candidate]]

# Module-level cache for collision candidates and ncon.
# Keyed by unique model identity (same approach as scan.py).
_collision_cache: dict[int, tuple["CandidateSet", int]] = {}
_collision_model_ids: dict[int, tuple[int, object]] = {}
_collision_model_id_counter = 0


def _collision_model_id(m) -> int:
    """Return a unique cache ID for this model instance."""
    global _collision_model_id_counter
    mid = id(m)
    stored = _collision_model_ids.get(mid)
    if stored is not None:
        cache_id, ref = stored
        if ref is m:
            return cache_id
    _collision_model_id_counter += 1
    cache_id = _collision_model_id_counter
    _collision_model_ids[mid] = (cache_id, m)
    return cache_id


def clear_collision_cache():
    """Clear precomputed collision caches (useful for testing)."""
    _collision_cache.clear()
    _collision_model_ids.clear()


# pair-wise collision functions
_COLLISION_FUNC = {
    (GeomType.PLANE, GeomType.SPHERE): plane_sphere,
    (GeomType.PLANE, GeomType.CAPSULE): plane_capsule,
    (GeomType.PLANE, GeomType.BOX): plane_convex,
    (GeomType.PLANE, GeomType.MESH): plane_convex,
    (GeomType.HFIELD, GeomType.SPHERE): hfield_sphere,
    (GeomType.HFIELD, GeomType.CAPSULE): hfield_capsule,
    (GeomType.HFIELD, GeomType.BOX): hfield_convex,
    (GeomType.HFIELD, GeomType.MESH): hfield_convex,
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
    key: tuple[GeomType | mujoco.mjtGeom, GeomType | mujoco.mjtGeom],
) -> Callable[[GeomInfo, GeomInfo], collision_types.Collision] | None:
    """Returns a collision function given a pair of geom types."""
    return _COLLISION_FUNC.get(key, None)


def _add_candidate(
    result: CandidateSet,
    m: Model | mujoco.MjModel,
    g1: int,
    g2: int,
    ipair: int = -1,
    geom_convex_data: tuple | None = None,
):
    """Adds a candidate to test for collision."""
    g1, g2, ipair = int(g1), int(g2), int(ipair)
    t1, t2 = int(m.geom_type[g1]), int(m.geom_type[g2])
    if t1 > t2:
        t1, t2, g1, g2 = t2, t1, g2, g1

    def mesh_key(i):
        if geom_convex_data is not None:
            convex_data = list(geom_convex_data)
        elif isinstance(m, Model):
            convex_data = [m.geom_convex_face, m.geom_convex_vert, m.geom_convex_edge]
        else:
            convex_data = [[None] * m.ngeom] * 3
        key = tuple((-1,) if v[i] is None else v[i].shape for v in convex_data)
        return key

    k1, k2 = mesh_key(g1), mesh_key(g2)

    candidates = {(c.geom1, c.geom2) for c in result.get((t1, t2, k1, k2), [])}
    if (g1, g2) in candidates:
        return

    if ipair > -1:
        candidate = Candidate(g1, g2, ipair, -1, int(m.pair_dim[ipair]))
    elif m.geom_priority[g1] != m.geom_priority[g2]:
        gp = g1 if m.geom_priority[g1] > m.geom_priority[g2] else g2
        candidate = Candidate(g1, g2, -1, int(gp), int(m.geom_condim[gp]))
    else:
        dim = int(max(m.geom_condim[g1], m.geom_condim[g2]))
        candidate = Candidate(g1, g2, -1, -1, dim)

    result.setdefault((t1, t2, k1, k2), []).append(candidate)


def _pair_params(
    m: Model,
    ipair: torch.Tensor,
) -> SolverParams:
    """Gets solver params for pair geoms."""
    friction = torch.clamp_min(m.pair_friction[ipair], mujoco.mjMINMU)
    solref = m.pair_solref[ipair]
    solreffriction = m.pair_solreffriction[ipair]
    solimp = m.pair_solimp[ipair]
    margin = m.pair_margin[ipair]
    gap = m.pair_gap[ipair]

    return SolverParams(
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        margin=margin,
        gap=gap,
        batch_size=[],
    )


def _priority_params(
    m: Model,
    geomp: torch.Tensor,
    geom_pairs: torch.Tensor,
) -> SolverParams:
    """Gets solver params from priority geoms."""
    friction = m.geom_friction[geomp][:, _FRICTION_IDX.get(torch.long, geomp.device)]
    solref = m.geom_solref[geomp]
    solreffriction = torch.zeros(geomp.shape + (mujoco.mjNREF,), dtype=solref.dtype, device=solref.device)
    solimp = m.geom_solimp[geomp]
    margin = torch.amax(m.geom_margin[geom_pairs.T], axis=0)
    gap = torch.amax(m.geom_gap[geom_pairs.T], axis=0)

    return SolverParams(
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        margin=margin,
        gap=gap,
        batch_size=[],
    )


def _dynamic_params(
    m: Model,
    g1: torch.Tensor,
    g2: torch.Tensor,
) -> SolverParams:
    """Gets solver params for dynamic geoms."""
    friction = torch.maximum(m.geom_friction[g1], m.geom_friction[g2])
    friction = friction[:, _FRICTION_IDX.get(torch.long, g1.device)]

    minval = _MINVAL.get(m.geom_solmix.dtype, g1.device)
    solmix1, solmix2 = m.geom_solmix[g1], m.geom_solmix[g2]
    mix = solmix1 / (solmix1 + solmix2)
    mix = torch.where((solmix1 < minval) & (solmix2 < minval), 0.5, mix)
    mix = torch.where((solmix1 < minval) & (solmix2 >= minval), 0.0, mix)
    mix_u = mix.unsqueeze(-1)

    solref1, solref2 = m.geom_solref[g1], m.geom_solref[g2]
    solref = torch.minimum(solref1, solref2)
    s_mix = mix_u * solref1 + (1 - mix_u) * solref2
    solref = torch.where((solref1[0] > 0) & (solref2[0] > 0), s_mix, solref)
    solreffriction = torch.zeros(g1.shape + (mujoco.mjNREF,), dtype=solref.dtype, device=solref.device)
    solimp = mix_u * m.geom_solimp[g1] + (1 - mix_u) * m.geom_solimp[g2]
    margin = torch.maximum(m.geom_margin[g1], m.geom_margin[g2])
    gap = torch.maximum(m.geom_gap[g1], m.geom_gap[g2])

    return SolverParams(
        friction=friction,
        solref=solref,
        solreffriction=solreffriction,
        solimp=solimp,
        margin=margin,
        gap=gap,
        batch_size=[],
    )


def _pair_info(
    m: Model,
    d: Data,
    g1: torch.Tensor,
    g2: torch.Tensor,
    geom1_list: Sequence[int],
    geom2_list: Sequence[int],
) -> tuple[GeomInfo, GeomInfo, int]:
    """Returns geom pair info for calculating collision."""
    n1, n2 = g1.shape[0], g2.shape[0]
    info1 = GeomInfo(
        pos=d.geom_xpos[g1],
        mat=d.geom_xmat[g1],
        geom_size=m.geom_size[g1],
        batch_size=[n1],
    )
    info2 = GeomInfo(
        pos=d.geom_xpos[g2],
        mat=d.geom_xmat[g2],
        geom_size=m.geom_size[g2],
        batch_size=[n2],
    )
    if m.geom_convex_face[geom1_list[0]] is not None:
        info1 = info1.replace(
            face=torch.stack([m.geom_convex_face[i] for i in geom1_list]),
            vert=torch.stack([m.geom_convex_vert[i] for i in geom1_list]),
            edge=torch.stack([m.geom_convex_edge[i] for i in geom1_list]),
            facenorm=torch.stack([m.geom_convex_facenormal[i] for i in geom1_list]),
        )
    if m.geom_convex_face[geom2_list[0]] is not None:
        info2 = info2.replace(
            face=torch.stack([m.geom_convex_face[i] for i in geom2_list]),
            vert=torch.stack([m.geom_convex_vert[i] for i in geom2_list]),
            edge=torch.stack([m.geom_convex_edge[i] for i in geom2_list]),
            facenorm=torch.stack([m.geom_convex_facenormal[i] for i in geom2_list]),
        )
    return info1, info2, 0


def _body_pair_filter(m: Model | mujoco.MjModel, b1: int, b2: int) -> bool:
    """Filters body pairs for collision."""
    dsbl_filterparent = m.opt.disableflags & DisableBit.FILTERPARENT
    weld1 = m.body_weldid[b1]
    weld2 = m.body_weldid[b2]
    parent_weld1 = m.body_weldid[m.body_parentid[weld1]]
    parent_weld2 = m.body_weldid[m.body_parentid[weld2]]

    if weld1 == weld2:
        # filter out self-collisions
        return True

    if not dsbl_filterparent and weld1 != 0 and weld2 != 0 and (weld1 == parent_weld2 or weld2 == parent_weld1):
        # filter out parent-child collisions
        return True

    return False


def _hfield_subgrid_size(m: Model, hfield_data_id: int, geom_rbound: float) -> tuple[int, int]:
    """Computes subgrid size for hfield collision based on object bounding radius."""
    hfield_size = m.hfield_size[hfield_data_id]
    nrow = int(m.hfield_nrow[hfield_data_id])
    ncol = int(m.hfield_ncol[hfield_data_id])
    xtick = 2.0 * float(hfield_size[0]) / (ncol - 1)
    ytick = 2.0 * float(hfield_size[1]) / (nrow - 1)
    xbound = int(_math.ceil(2 * geom_rbound / xtick)) + 1
    xbound = min(xbound, ncol)
    ybound = int(_math.ceil(2 * geom_rbound / ytick)) + 1
    ybound = min(ybound, nrow)
    return (xbound, ybound)


@torch.compiler.disable
def _collide_hfield_geoms(
    m: Model,
    d: Data,
    candidates: Sequence[Candidate],
    fn: Callable,
) -> Contact:
    """Collides hfield geom pairs individually (no vmap)."""
    typ_cands: dict[tuple[bool, bool], list[Candidate]] = {}
    for c in candidates:
        typ = (c.ipair > -1, c.geomp > -1)
        typ_cands.setdefault(typ, []).append(c)

    geom1_ids: list[int] = []
    geom2_ids: list[int] = []
    dims: list[int] = []
    params: list[SolverParams] = []
    all_dist: list[torch.Tensor] = []
    all_pos: list[torch.Tensor] = []
    all_frame: list[torch.Tensor] = []

    for (pair, priority), cands in typ_cands.items():
        if pair:
            p = _pair_params(m, torch.tensor([c.ipair for c in cands]))
        elif priority:
            p = _priority_params(
                m,
                torch.tensor([c.geomp for c in cands]),
                torch.tensor([(c.geom1, c.geom2) for c in cands]),
            )
        else:
            p = _dynamic_params(
                m,
                torch.tensor([c.geom1 for c in cands]),
                torch.tensor([c.geom2 for c in cands]),
            )
        params.append(p)

        for c in cands:
            geom1_ids.append(c.geom1)
            geom2_ids.append(c.geom2)
            dims.append(c.dim)

            g1, g2 = c.geom1, c.geom2
            hfield_data_id = int(m.geom_dataid[g1])
            h_info = mesh_module.hfield(m, hfield_data_id)
            h_info = h_info.replace(
                pos=d.geom_xpos[g1].to(torch.float64),
                mat=d.geom_xmat[g1].to(torch.float64),
            )

            obj_info = GeomInfo(
                pos=d.geom_xpos[g2],
                mat=d.geom_xmat[g2],
                geom_size=m.geom_size[g2],
                batch_size=[],
            )
            if m.geom_convex_face[g2] is not None:
                obj_info = obj_info.replace(
                    face=m.geom_convex_face[g2],
                    vert=m.geom_convex_vert[g2],
                    edge=m.geom_convex_edge[g2],
                    facenorm=m.geom_convex_facenormal[g2],
                )

            rbound = float(m.geom_rbound_hfield[g2])
            subgrid_size = _hfield_subgrid_size(m, hfield_data_id, rbound)
            dist_i, pos_i, frame_i = fn(h_info, obj_info, subgrid_size)
            all_dist.append(dist_i)
            all_pos.append(pos_i)
            all_frame.append(frame_i)

    dist = torch.cat(all_dist)
    pos = torch.cat(all_pos)
    frame = torch.cat(all_frame)
    ncon_per_pair = fn.ncon

    def _concat(*x):
        return np.concatenate(x, axis=0) if isinstance(x[0], np.ndarray) else torch.cat(x, dim=0)

    params = torch.utils._pytree.tree_map(_concat, *params) if len(params) > 1 else params[0]
    geom1_t = torch.tensor(geom1_ids, device=dist.device)
    geom2_t = torch.tensor(geom2_ids, device=dist.device)
    contact_dim = torch.tensor(dims, dtype=torch.int32, device=dist.device)
    n_repeat = ncon_per_pair
    geom1_t, geom2_t, contact_dim, params = torch.utils._pytree.tree_map(
        lambda x: x.repeat_interleave(n_repeat, dim=0),
        (geom1_t, geom2_t, contact_dim, params),
    )

    return Contact(
        dist=dist,
        pos=pos,
        frame=frame,
        includemargin=params.margin - params.gap,
        friction=params.friction,
        solref=params.solref,
        solreffriction=params.solreffriction,
        solimp=params.solimp,
        contact_dim=contact_dim,
        geom1=geom1_t,
        geom2=geom2_t,
        geom=torch.stack([geom1_t, geom2_t], dim=-1),
        efc_address=torch.full((dist.shape[0],), -1, dtype=torch.int64, device=dist.device),
        batch_size=[dist.shape[0]],
    )


def precompute_collision_indices(candidates: Sequence[Candidate]) -> dict:
    """Pre-compute index tensors for a collision group at device_put time.

    Groups candidates by type (pair/priority/dynamic) and builds all index
    tensors wrapped as _DeviceCachedTensor for lazy device transfer.
    """
    typ_cands: dict[tuple[bool, bool], list[Candidate]] = {}
    for c in candidates:
        typ = (c.ipair > -1, c.geomp > -1)
        typ_cands.setdefault(typ, []).append(c)

    geom1_list: list[int] = []
    geom2_list: list[int] = []
    dims_list: list[int] = []
    param_groups: list[tuple[Callable, dict[str, _DeviceCachedTensor]]] = []

    for (pair, priority), cands in typ_cands.items():
        geom1_list.extend([c.geom1 for c in cands])
        geom2_list.extend([c.geom2 for c in cands])
        dims_list.extend([c.dim for c in cands])
        if pair:
            param_groups.append(
                (
                    _pair_params,
                    {
                        "ipair": _DeviceCachedTensor(torch.tensor([c.ipair for c in cands])),
                    },
                )
            )
        elif priority:
            param_groups.append(
                (
                    _priority_params,
                    {
                        "geomp": _DeviceCachedTensor(torch.tensor([c.geomp for c in cands])),
                        "geom_pairs": _DeviceCachedTensor(torch.tensor([(c.geom1, c.geom2) for c in cands])),
                    },
                )
            )
        else:
            param_groups.append(
                (
                    _dynamic_params,
                    {
                        "g1": _DeviceCachedTensor(torch.tensor([c.geom1 for c in cands])),
                        "g2": _DeviceCachedTensor(torch.tensor([c.geom2 for c in cands])),
                    },
                )
            )

    return {
        "geom1_t": _DeviceCachedTensor(torch.tensor(geom1_list, dtype=torch.long)),
        "geom2_t": _DeviceCachedTensor(torch.tensor(geom2_list, dtype=torch.long)),
        "contact_dim_t": _DeviceCachedTensor(torch.tensor(dims_list, dtype=torch.int32)),
        "geom1_list": tuple(geom1_list),
        "geom2_list": tuple(geom2_list),
        "param_groups": tuple(param_groups),
    }


def _collide_geoms(
    m: Model,
    d: Data,
    geom_types: tuple[GeomType, GeomType],
    candidates: Sequence[Candidate],
    fn: Callable | None = None,
    precomp: dict | None = None,
) -> Contact:
    """Collides a geom pair."""
    if fn is None:
        fn = get_collision_fn(geom_types)
    device = d.geom_xpos.device

    if not fn:
        return Contact.zero(device=device)

    if geom_types[0] == GeomType.HFIELD:
        return _collide_hfield_geoms(m, d, candidates, fn)

    geom1_t = precomp["geom1_t"]
    geom2_t = precomp["geom2_t"]
    contact_dim = precomp["contact_dim_t"]

    params = []
    for params_fn, indices in precomp["param_groups"]:
        params.append(params_fn(m, **indices))

    g1, g2, in_axes = _pair_info(
        m,
        d,
        geom1_t,
        geom2_t,
        precomp["geom1_list"],
        precomp["geom2_list"],
    )
    res = torch.vmap(fn, in_axes)(g1, g2)
    dist, pos, frame = res

    ncon_per_pair = dist.shape[1] if dist.ndim > 1 else 1
    dist = dist.reshape(-1)
    pos = pos.reshape(-1, 3)
    frame = frame.reshape(-1, 3, 3)

    def _concat(*x):
        return np.concatenate(x, axis=0) if isinstance(x[0], np.ndarray) else torch.cat(x, dim=0)

    params = torch.utils._pytree.tree_map(_concat, *params) if len(params) > 1 else params[0]
    n_repeat = ncon_per_pair
    geom1_t, geom2_t, contact_dim, params = torch.utils._pytree.tree_map(
        lambda x: x.repeat_interleave(n_repeat, dim=0),
        (geom1_t, geom2_t, contact_dim, params),
    )

    return Contact(
        dist=dist,
        pos=pos,
        frame=frame,
        includemargin=params.margin - params.gap,
        friction=params.friction,
        solref=params.solref,
        solreffriction=params.solreffriction,
        solimp=params.solimp,
        contact_dim=contact_dim,
        geom1=geom1_t,
        geom2=geom2_t,
        geom=torch.stack([geom1_t, geom2_t], dim=-1),
        efc_address=torch.full((dist.shape[0],), -1, dtype=torch.int64, device=dist.device),
        batch_size=[dist.shape[0]],
    )


def _max_contact_points(m: Model) -> int:
    """Returns the maximum number of contact points when set as a numeric."""
    for i in range(m.nnumeric):
        name = m.names[m.name_numericadr[i] :].decode("utf-8").split("\x00", 1)[0]
        if name == "max_contact_points":
            return int(m.numeric_data[m.numeric_adr[i]])

    return -1


def collision_candidates(
    m: Model | mujoco.MjModel,
    geom_convex_data: tuple | None = None,
) -> CandidateSet:
    """Returns candidates for collision checking."""
    candidate_set = {}

    for ipair in range(m.npair):
        g1, g2 = m.pair_geom1[ipair], m.pair_geom2[ipair]
        _add_candidate(candidate_set, m, g1, g2, ipair, geom_convex_data=geom_convex_data)

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
                    _add_candidate(candidate_set, m, g1, g2, geom_convex_data=geom_convex_data)

    return candidate_set


def make_condim(m: Model | mujoco.MjModel) -> torch.Tensor:
    """Returns per-contact condim array, sorted ascending (1s before 3s).

    This is computed purely from Model (no Data needed) and mirrors MJX's
    ``make_condim`` in ``mujoco/mjx/_src/collision_driver.py``.
    """
    if m.opt.disableflags & DisableBit.CONTACT:
        return torch.empty(0, dtype=torch.long)

    candidates = collision_candidates(m)
    max_count = _max_contact_points(m)

    dims: list = []
    for k, v in candidates.items():
        fn = get_collision_fn(k[0:2])
        if fn is None:
            continue
        ncon_per_pair = fn.ncon  # pytype: disable=attribute-error
        for c in v:
            dims.extend([c.dim] * ncon_per_pair)

    dims.sort()

    if max_count > -1 and len(dims) > max_count:
        dims = dims[:max_count]

    return torch.tensor(dims, dtype=torch.long) if dims else torch.empty(0, dtype=torch.long)


def ncon(m: Model) -> int:
    """Returns the number of contacts computed in MJX given a model."""
    return make_condim(m).numel()


def _get_collision_cache(m: Model) -> tuple:
    """Get or build cached collision candidates and ncon for a model.

    Returns a tuple of (collision_groups, ncon, max_contact_points) where
    collision_groups is a list of (fn, geom_types, candidates) tuples with the
    collision function already resolved.  This avoids dict lookups with numpy
    keys at runtime which break under torch.compile.
    """
    cache_id = _collision_model_id(m)
    cached = _collision_cache.get(cache_id)
    if cached is not None:
        return cached

    candidate_set = collision_candidates(m)
    ncon_ = ncon(m)
    max_cp = _max_contact_points(m)

    # Pre-resolve collision functions using proper GeomType enums.
    collision_groups = []
    for key, candidates in candidate_set.items():
        geom_types = (GeomType(int(key[0])), GeomType(int(key[1])))
        fn = get_collision_fn(geom_types)
        collision_groups.append((fn, geom_types, candidates))

    result = (collision_groups, ncon_, max_cp)
    _collision_cache[cache_id] = result
    return result


def constraint_sizes(m: Model) -> tuple[int, int, int, int, int]:
    """Return (ne, nf, nl, ncon, nefc) purely from Model (no Data needed).

    These are pre-computed at ``device_put`` time and stored on the Model
    so that the values are plain Python ints visible as compile-time
    constants to ``torch.compile``.
    """
    return m.constraint_sizes_py


def collision(m: Model, d: Data) -> Data:
    """Collides geometries."""
    collision_groups = m._device_precomp["collision_groups_py"]
    ncon_ = m.constraint_sizes_py[3]
    max_cp = m.collision_max_cp_py
    total = m.collision_total_contacts_py

    if ncon_ == 0:
        d.update_(contact=Contact.zero(device=d.qpos.device), ncon=torch.zeros((), dtype=torch.int32, device=d.qpos.device))
        return d

    contacts = []
    for fn, geom_types, candidates, precomp in collision_groups:
        contacts.append(_collide_geoms(m, d, geom_types, candidates, fn=fn, precomp=precomp))

    # Concatenate all contacts.
    contact = torch.cat(contacts)

    if max_cp > -1 and total > max_cp:
        # get top-k contacts
        _, idx = torch.topk(-contact.dist, k=max_cp)
        contact = contact[idx]

    # Sort contacts by condim (1s before 3s) for consistent constraint ordering.
    sort_idx = torch.argsort(contact.contact_dim)
    contact = contact[sort_idx]

    # Compute efc_address with variable strides per condim:
    # condim=1 produces 1 constraint row, condim=3 produces 4 (pyramidal).
    ne, nf, nl, _, _ = constraint_sizes(m)
    ns = ne + nf + nl
    dims_t = contact.contact_dim
    rows_per_contact = torch.where(dims_t == 1, 1, (dims_t - 1) * 2)
    zeros = torch.zeros(1, dtype=rows_per_contact.dtype, device=rows_per_contact.device)
    offsets = torch.cumsum(torch.cat([zeros, rows_per_contact[:-1]]), dim=0)
    contact = contact.replace(
        efc_address=(ns + offsets).to(torch.int64),
    )

    d.update_(contact=contact, ncon=torch.full((), ncon_, dtype=torch.int32, device=contact.dist.device))
    return d
