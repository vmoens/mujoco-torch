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
"""Scan across data ordered by body joint types and kinematic tree order."""

import dataclasses
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch
from tensordict import UnbatchedTensor
from torch.utils._pytree import tree_map

from mujoco_torch._src.math import concatenate
from mujoco_torch._src.types import JointType, Model, TrnType

Y = TypeVar("Y")

# Module-level caches for scan grouping indices.  Keyed by
# (cache_id, in_types, out_types, group_by_or_reverse).
# Models carry a ``cache_id`` int assigned at device_put time.
_flat_cache: dict = {}
_body_tree_cache: dict = {}


def _model_cache_id(m: Model) -> int:
    """Return the unique cache ID for this model (set at device_put time)."""
    return m.cache_id


def clear_scan_caches():
    """Clear precomputed scan grouping caches (useful for testing)."""
    _flat_cache.clear()
    _body_tree_cache.clear()


def _resolve_cached_tensors(obj, device: torch.device):
    """Recursively replace _DeviceCachedTensor and CPU tensors with device tensors.

    Returns a new structure where every _DeviceCachedTensor has been
    resolved to a regular torch.Tensor on *device* and every plain CPU
    tensor has been moved, so that ``torch.compile`` never sees a
    CPU→GPU copy inside the traced graph.
    """
    if isinstance(obj, _DeviceCachedTensor):
        return obj.to(device)
    if isinstance(obj, torch.Tensor) and obj.device != device:
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _resolve_cached_tensors(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(_resolve_cached_tensors(v, device) for v in obj)
    if isinstance(obj, list):
        return [_resolve_cached_tensors(v, device) for v in obj]
    return obj


def warm_device_caches(cache_id: int, device: torch.device):
    """Resolve all _DeviceCachedTensor entries for *cache_id* to *device*.

    Call this after ``Model.to(device)`` so that ``torch.compile`` never
    sees a CPU→GPU copy inside the traced graph.  Replaces
    _DeviceCachedTensor instances with plain device tensors in-place.
    """
    device = torch.device(device)

    for key in list(_flat_cache):
        if key[0] == cache_id:
            _flat_cache[key] = _resolve_cached_tensors(_flat_cache[key], device)
    for key in list(_body_tree_cache):
        if key[0] == cache_id:
            _body_tree_cache[key] = _resolve_cached_tensors(
                _body_tree_cache[key],
                device,
            )


# Registry of all known scan call-site signatures.
# Each entry: (in_types, out_types, group_by_or_reverse,
#              [(arg_position, model_field_name), ...])
# The last element lists which arg positions are static model fields.
_KNOWN_FLAT_CALLS = [
    ("jbbjj", "v", "j", [(0, "jnt_type")]),
    ("uuua", "a", "u", [(0, "actuator_dyntype")]),
    ("uuuuuuuuu", "u", "u", [(0, "actuator_gaintype"), (2, "actuator_biastype")]),
    ("uuaau", "a", "u", [(0, "actuator_dyntype")]),
    ("au", "a", "u", []),
    ("jqv", "q", "j", [(0, "jnt_type")]),
    ("jjqq", "v", "j", [(0, "jnt_type")]),
]

_KNOWN_BODY_TREE_CALLS = [
    ("jjjqqbb", "qjjbbb", False, [(0, "jnt_type")]),
    ("bb", "bb", True, []),
    ("b", "b", True, []),
    ("jvv", "bv", False, [(0, "jnt_type")]),
    ("vvvv", "b", False, []),
]


def _signature_applies(m, in_types, out_types, group_by=None):
    """Check whether a scan signature is relevant for a given model."""
    all_types = set(in_types + out_types)
    if group_by is not None:
        all_types.add(group_by)
    if ("u" in all_types or "a" in all_types) and m.nu == 0:
        return False
    if ("j" in all_types or "q" in all_types or "v" in all_types) and m.njnt == 0:
        return False
    return True


def precompute_scan_caches(m, cache_id: int):
    """Pre-build all scan caches for known call-site signatures.

    Called from ``device_put`` so that ``flat()`` and ``body_tree()``
    never need to build caches at runtime (which would require
    ``@torch.compiler.disable``).
    """
    for in_types, out_types, group_by, static_fields in _KNOWN_FLAT_CALLS:
        if not _signature_applies(m, in_types, out_types, group_by):
            continue
        cache_key = (cache_id, in_types, out_types, group_by)
        cache = _build_flat_cache(m, in_types, out_types, group_by)
        # Pre-extract static args for all groups
        key_typ_ids_torch = cache["key_typ_ids_torch"]
        group_has_output = cache["group_has_output"]
        pre_extracted = []
        for (_, typ_ids_t), has_output in zip(key_typ_ids_torch, group_has_output):
            if not has_output:
                pre_extracted.append(None)
                continue
            group_static = [None] * len(in_types)
            for pos, field_name in static_fields:
                static_array = getattr(m, field_name)
                typ = in_types[pos]
                subset = _take(static_array, typ_ids_t[typ])
                group_static[pos] = _validate_and_convert_subset(subset)
            pre_extracted.append(group_static)
        cache["pre_extracted_static"] = pre_extracted
        _flat_cache[cache_key] = cache

    for in_types, out_types, reverse, static_fields in _KNOWN_BODY_TREE_CALLS:
        if not _signature_applies(m, in_types, out_types):
            continue
        cache_key = (cache_id, in_types, out_types, reverse)
        cache = _build_body_tree_cache(m, in_types, out_types, reverse)
        # Pre-extract static args for all groups
        key_in_take_torch = cache["key_in_take_torch"]
        keys = cache["keys"]
        pre_extracted = {}
        for key in keys:
            group_static = [None]  # carry is never static
            for i, ids in enumerate(key_in_take_torch[key]):
                static_val = None
                for pos, field_name in static_fields:
                    if pos == i:
                        static_array = getattr(m, field_name)
                        subset = _take(static_array, ids)
                        static_val = _validate_and_convert_subset(subset)
                        break
                group_static.append(static_val)
            pre_extracted[key] = group_static
        cache["pre_extracted_static"] = pre_extracted
        _body_tree_cache[cache_key] = cache


def _np_to_long(x):
    """Convert numpy array to torch.LongTensor."""
    return torch.as_tensor(np.asarray(x), dtype=torch.long)


class _DeviceCachedTensor:
    """A CPU index tensor that lazily caches per-device copies.

    On first access for a given device, moves the tensor and caches it.
    Subsequent accesses for the same device return the cached copy with
    no CPU-GPU transfer.
    """

    def __init__(self, cpu_tensor: torch.Tensor):
        self._cpu = cpu_tensor
        self._cache: dict[torch.device, torch.Tensor] = {}

    def to(self, device: torch.device) -> torch.Tensor:
        t = self._cache.get(device)
        if t is None:
            t = self._cpu.to(device)
            self._cache[device] = t
        return t

    @property
    def device(self):
        return self._cpu.device

    def __getattr__(self, name):
        return getattr(self._cpu, name)

    def __len__(self):
        return len(self._cpu)


def _cached_long(x) -> _DeviceCachedTensor:
    """Convert to a device-cached LongTensor (lazy GPU transfer)."""
    cpu_tensor = torch.as_tensor(np.asarray(x), dtype=torch.long).cpu()
    return _DeviceCachedTensor(cpu_tensor)


def _cat_device_safe(tensors):
    """Concatenates tensors, moving all to the device of the first non-CPU tensor."""
    device = None
    for t in tensors:
        if t.device.type != "cpu":
            device = t.device
            break
    if device is not None:
        tensors = tuple(t.to(device) if t.device != device else t for t in tensors)
    return torch.cat(tensors)


def _take(obj: Y, idx) -> Y:
    """Takes idxs on any pytree given to it.

    Supports both numpy and torch index arrays.

    Args:
      obj: an input pytree
      idx: indices to take (numpy array or torch.LongTensor)

    Returns:
      obj pytree with leaves taken by idxs
    """
    if isinstance(obj, np.ndarray):
        if isinstance(idx, _DeviceCachedTensor):
            return obj[idx._cpu.numpy()]
        if isinstance(idx, torch.Tensor):
            return obj[idx.cpu().numpy()]
        return obj[idx]

    # Resolve idx to a regular tensor once (or keep _DeviceCachedTensor for
    # lazy device resolution — needed for CUDA graph capture).
    if isinstance(idx, _DeviceCachedTensor):
        cached = idx

        def take(x):
            if not x.shape[0]:
                return x
            return x[cached.to(x.device)].clone()

        return tree_map(take, obj)

    if not isinstance(idx, torch.Tensor):
        idx = _np_to_long(idx)

    def take(x):
        if not x.shape[0]:
            return x
        i = idx.to(x.device) if idx.device != x.device else idx
        return x[i].clone()

    return tree_map(take, obj)


def _as_numpy(val):
    """Convert to numpy, handling UnbatchedTensor and regular tensors."""
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return val


def _static_arg_mask(m: Model, args: tuple) -> list[bool]:
    """Determine which args are static metadata (should not go through vmap).

    Uses both isinstance checks (for np.ndarray / UnbatchedTensor) and
    Model type annotations (matching arg identity against UnbatchedTensor
    fields).
    """
    unbatched_ids = {id(getattr(m, f.name)) for f in dataclasses.fields(type(m)) if f.type is UnbatchedTensor}
    return [isinstance(a, (np.ndarray, UnbatchedTensor)) or id(a) in unbatched_ids for a in args]


def _q_bodyid(m: Model) -> np.ndarray:
    """Returns the bodyid for each qpos adress."""
    jnt_type = _as_numpy(m.jnt_type)
    q_bodyids = [np.array([], dtype=np.int32)]
    for jt, jnt_bodyid in zip(jnt_type, m.jnt_bodyid):
        width = {JointType.FREE: 7, JointType.BALL: 4}.get(jt, 1)
        q_bodyids.append(np.repeat(jnt_bodyid, width))
    return np.concatenate(q_bodyids)


def _q_jointid(m: Model) -> np.ndarray:
    """Returns the jointid for each qpos adress."""
    jnt_type = _as_numpy(m.jnt_type)
    q_jointid = [np.array([], dtype=np.int32)]
    for i, jt in enumerate(jnt_type):
        width = {JointType.FREE: 7, JointType.BALL: 4}.get(jt, 1)
        q_jointid.append(np.repeat(i, width))
    return np.concatenate(q_jointid)


def _index(haystack, needle):
    """Returns indexes in haystack for elements in needle."""
    idx = np.argsort(haystack)
    sorted_haystack = haystack[idx]
    sorted_idx = np.searchsorted(sorted_haystack, needle)
    idx = np.take(idx, sorted_idx, mode="clip")
    idx[haystack[idx] != needle] = -1
    return idx


def _to_safe(val):
    """Convert a representative value to a Dynamo-safe Python/tensor form.

    Handles both numpy arrays and torch tensors (from UnbatchedTensor).
    """
    if isinstance(val, torch.Tensor):
        if val.ndim == 0:
            return val.item()
        if val.numel() == 0:
            return ()
        if val.ndim == 1 and val.dtype in (torch.int32, torch.int64, torch.bool):
            return tuple(int(x) for x in val.tolist())
        return val
    if isinstance(val, (np.integer, np.floating, np.bool_)):
        return val.item()
    if not isinstance(val, np.ndarray):
        return val
    if val.ndim == 0:
        return val.item()
    if val.size == 0:
        return ()
    if val.ndim == 1 and isinstance(val.flat[0], (np.integer, np.bool_)):
        return tuple(int(x) for x in val)
    return torch.as_tensor(np.ascontiguousarray(val))


def _validate_and_convert_subset(subset):
    """Validate that a per-group subset has identical rows, then convert."""
    if isinstance(subset, torch.Tensor):
        if subset.shape[0] > 0 and not torch.all(subset == subset[0]):
            raise RuntimeError(f"static arg elements do not match: {subset}")
    else:
        if subset.shape[0] > 0 and not np.all(subset == subset[0]):
            raise RuntimeError(f"static arg elements do not match: {subset}")
    val = subset[0] if subset.shape[0] > 0 else subset
    return _to_safe(val)


def _extract_static_for_flat(args, in_types, key_typ_ids_torch, group_has_output, is_static):
    """Extract static args for all flat-scan groups at once.

    This is only called as a fallback for signatures not registered in
    ``_KNOWN_FLAT_CALLS``.  For known signatures, pre-extracted values are
    stored in the cache at ``device_put`` time.
    """
    result = []
    for (_, typ_ids_t), has_output in zip(key_typ_ids_torch, group_has_output):
        if not has_output:
            result.append(None)
            continue
        group_static = []
        for arg, typ, static in zip(args, in_types, is_static):
            if not static:
                group_static.append(None)
                continue
            subset = _take(arg, typ_ids_t[typ])
            group_static.append(_validate_and_convert_subset(subset))
        result.append(group_static)
    return result


def _extract_static_for_body_tree(args, key_in_take_torch, keys, is_static):
    """Extract static args for all body-tree groups at once.

    This is only called as a fallback for signatures not registered in
    ``_KNOWN_BODY_TREE_CALLS``.  For known signatures, pre-extracted values
    are stored in the cache at ``device_put`` time.
    """
    result = {}
    for key in keys:
        group_static = [None]  # carry is never static
        for arg, ids, static in zip(args, key_in_take_torch[key], is_static):
            if not static:
                group_static.append(None)
                continue
            subset = _take(arg, ids)
            group_static.append(_validate_and_convert_subset(subset))
        result[key] = group_static
    return result


def _invert_segment_ids(segment_ids, num_segments):
    """Precompute gather indices for scatter_add-free segment sum.

    Returns:
      inv_idx: (num_segments, max_children) LongTensor of source indices
      inv_mask: (num_segments, max_children) float mask (1.0 or 0.0)
    """
    buckets = [[] for _ in range(num_segments)]
    for i, s in enumerate(segment_ids):
        buckets[s].append(i)
    max_children = max((len(b) for b in buckets), default=0)
    inv_idx = np.zeros((num_segments, max_children), dtype=np.int64)
    inv_mask = np.zeros((num_segments, max_children), dtype=np.float64)
    for s, bucket in enumerate(buckets):
        for j, idx in enumerate(bucket):
            inv_idx[s, j] = idx
            inv_mask[s, j] = 1.0
    return inv_idx, torch.tensor(inv_mask)


def _gather_segment_sum(data, inv_idx, inv_mask):
    """Segment sum via gather + masked reduction (no atomics)."""
    idx = inv_idx.to(data.device)
    mask = inv_mask.to(data.device).to(data.dtype)
    gathered = data[idx]
    for _ in range(data.ndim - 1):
        mask = mask.unsqueeze(-1)
    return (gathered * mask).sum(1)


def _nvmap(f: Callable[..., Y], static_args, *args) -> Y:
    """A vmap that accepts pre-extracted static args.

    Args:
      f: function to be mapped over
      static_args: pre-extracted static args (from _extract_static_for_*)
      *args: tensor args to be vmapped

    Returns:
      the result of vmapping f over args
    """

    # remove empty args that we should not vmap over
    def _check_empty(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor) and a.shape[0] == 0:
            return None
        return a

    args = [_check_empty(a) for a in args]

    # torch.vmap does not accept None arguments, so filter them out and
    # reconstruct the full args list inside the vmapped function.
    none_mask = [a is None for a in args]
    vmap_args = [a for a in args if a is not None]

    def _ensure_tensor(x):
        """Convert numpy scalars/arrays and Python scalars to tensors for vmap compatibility."""
        if isinstance(x, (np.ndarray, np.integer, np.floating)):
            return torch.as_tensor(x)
        if isinstance(x, (int, float, bool)):
            return torch.as_tensor(x)
        return x

    def outer_f(*vmap_inputs, _static_args=static_args, _none_mask=none_mask):
        it = iter(vmap_inputs)
        full_args = [None if m else next(it) for m in _none_mask]
        full_args = [a if n is None else n for n, a in zip(_static_args, full_args)]
        result = f(*full_args)
        return tree_map(_ensure_tensor, result)

    if not vmap_args:
        return f(*[n if n is not None else None for n in static_args])

    if torch.compiler.is_compiling():
        return torch.vmap(outer_f, in_dims=tuple([0] * len(vmap_args)))(*vmap_args)

    try:
        return torch.vmap(outer_f, in_dims=tuple([0] * len(vmap_args)))(*vmap_args)
    except ValueError as e:
        if "NoneType" in str(e):
            return None
        raise


def _check_input(m: Model, args: Any, in_types: str) -> None:
    """Checks that scan input has the right shape."""
    if torch.compiler.is_compiling():
        return
    size = {
        "b": m.nbody,
        "j": m.njnt,
        "q": m.nq,
        "v": m.nv,
        "u": m.nu,
        "a": m.na,
        "s": m.nsite,
        "c": m.ncam,
    }
    for idx, (arg, typ) in enumerate(zip(args, in_types)):
        arg_len = len(arg)
        if arg_len != size[typ]:
            raise IndexError(
                f'f argument "{idx}" with type "{typ}" has length "{arg_len}"'
                f" which does not match the in_types[{idx}] expected length of "
                f'"{size[typ]}".'
            )


def _check_output(y: torch.Tensor, take_ids, typ: str, idx: int) -> None:
    """Checks that scan output has the right shape."""
    if torch.compiler.is_compiling():
        return
    n = take_ids.shape[0] if isinstance(take_ids, (np.ndarray, torch.Tensor)) else len(take_ids)
    if y.shape[0] != n:
        raise IndexError(
            f'f output "{idx}" with type "{typ}" has shape "{y.shape[0]}" '
            f"which does not match the out_types[{idx}] expected size of"
            f' "{n}".'
        )


def _build_flat_cache(m, in_types, out_types, group_by):
    """Precompute grouping indices for scan.flat (numpy-heavy, runs once)."""

    if group_by not in {"j", "u", "c"}:
        raise NotImplementedError(f'group by type "{group_by}" not implemented.')

    # Convert UnbatchedTensor fields to numpy for grouping logic
    jnt_type_np = _as_numpy(m.jnt_type)
    act_biastype_np = _as_numpy(m.actuator_biastype) if m.nu else None
    act_gaintype_np = _as_numpy(m.actuator_gaintype) if m.nu else None
    act_dyntype_np = _as_numpy(m.actuator_dyntype) if m.nu else None

    def key_j(type_ids):
        if any(t in "jqv" for t in in_types + out_types):
            return tuple(jnt_type_np[type_ids["j"]])
        return ()

    def type_ids_j(m, i):
        return {
            "b": i,
            "j": np.nonzero(m.jnt_bodyid == i)[0],
            "v": np.nonzero(m.dof_bodyid == i)[0],
            "q": np.nonzero(_q_bodyid(m) == i)[0],
        }

    def key_u(type_ids):
        ids_u, ids_j = type_ids["u"], type_ids["j"]
        return (
            act_biastype_np[ids_u],
            act_gaintype_np[ids_u],
            act_dyntype_np[ids_u],
            m.actuator_trntype[ids_u],
            jnt_type_np[ids_j],
            m.actuator_trnid[ids_u, 1] == -1,
        )

    def type_ids_u(m, i):
        typ_ids = {
            "u": i,
            "a": m.actuator_actadr[i],
            "j": (m.actuator_trnid[i, 0] if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT) else -1),
            "s": (m.actuator_trnid[i] if m.actuator_trntype[i] == TrnType.SITE else np.array([-1, -1])),
        }
        v, q = np.array([-1]), np.array([-1])
        if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT):
            v = np.nonzero(m.dof_jntid == typ_ids["j"])[0]
            q = np.nonzero(_q_jointid(m) == typ_ids["j"])[0]
        typ_ids.update({"v": v, "q": q})
        return typ_ids

    def key_c(type_ids):
        return m.cam_mode[type_ids["c"]], m.cam_targetbodyid[type_ids["c"]] >= 0

    def type_ids_c(unused_m, i):
        return {"c": i}

    type_ids_fn = {"j": type_ids_j, "u": type_ids_u, "c": type_ids_c}[group_by]
    key_fn = {"j": key_j, "u": key_u, "c": key_c}[group_by]

    all_types = set(in_types + out_types)
    n_items = {"j": m.nbody, "u": m.nu, "c": m.ncam}[group_by]
    key_typ_ids, order = {}, []
    for i in np.arange(n_items, dtype=np.int32):
        typ_ids = type_ids_fn(m, i)
        key = key_fn(typ_ids)
        order.append((key, typ_ids))
        for t in all_types:
            out = key_typ_ids.setdefault(key, {})
            val = np.expand_dims(typ_ids[t], axis=0)
            out[t] = np.concatenate((out[t], val)) if t in out else val

    key_typ_ids = list(sorted(key_typ_ids.items()))

    # Determine which groups could possibly have output (non-empty take ids)
    group_has_output = [any(typ_ids[v].size > 0 for v in out_types) for _, typ_ids in key_typ_ids]

    flat_ = {"j": "b", "u": "uaj", "c": "c"}[group_by]

    # ---- Precompute torch indices for runtime use ----

    # Convert typ_ids to device-cached LongTensors for _take operations
    key_typ_ids_torch = []
    for key, typ_ids in key_typ_ids:
        typ_ids_t = {t: _cached_long(v) for t, v in typ_ids.items()}
        key_typ_ids_torch.append((key, typ_ids_t))

    # Pre-compute reorder indices for the compile path.
    # Assumes all has_output groups produce results (always true in production;
    # only test callbacks that return None break this assumption).
    active_kti_np = [(k, v) for (k, v), ho in zip(key_typ_ids, group_has_output) if ho]
    active_keys = set(k for k, _ in active_kti_np)
    active_order = [typ_ids for key, typ_ids in order if key in active_keys]
    active_order_per_type = [[o[t] for o in active_order] for t in all_types]
    active_order_per_type = [
        (np.concatenate(o) if isinstance(o[0], np.ndarray) else np.array(o)) if o else np.array([], dtype=np.int64)
        for o in active_order_per_type
    ]
    order_dict = dict(zip(all_types, active_order_per_type))

    reorder_indices = {}
    for typ in set(out_types):
        ids = np.concatenate([np.hstack(v[typ]) for _, v in active_kti_np])
        input_order = order_dict[typ][np.where(order_dict[typ] != -1)]
        reorder_indices[typ] = _cached_long(_index(ids, input_order))

    return {
        "key_typ_ids": key_typ_ids,
        "key_typ_ids_torch": key_typ_ids_torch,
        "group_has_output": group_has_output,
        "flat_": flat_,
        "reorder_indices": reorder_indices,
    }


def _get_flat_cache(m, in_types, out_types, group_by):
    """Get or build flat scan cache for the given model and type signature.

    For known signatures (registered in ``_KNOWN_FLAT_CALLS``), the cache is
    pre-built at ``device_put`` time via ``precompute_scan_caches`` so this
    is a simple dict lookup with no graph break.
    """
    cache_key = (_model_cache_id(m), in_types, out_types, group_by)
    cache = _flat_cache.get(cache_key)
    if cache is None:
        cache = _build_flat_cache(m, in_types, out_types, group_by)
        _flat_cache[cache_key] = cache
    return cache


def flat(
    m: Model,
    f: Callable[..., Y],
    in_types: str,
    out_types: str,
    *args,
    group_by: str = "j",
) -> Y:
    r"""Scan a function across bodies or actuators.

    Scan group data according to type and batch shape then calls vmap(f) on it.
    Grouping indices are precomputed and cached per model to avoid redundant
    numpy work on repeated calls.

    Args:
      m: an mjx model
      f: a function to be scanned with the following type signature:
          def f(key, *args) -> y
        where
          ``key`` gives grouping key for this function instance
          ``*args`` are input arguments with types matching ``in_types``
          ``y`` is an output arguments with types matching ``out_type``
      in_types: string specifying the type of each input arg:
        'b': split according to bodies
        'j': split according to joint types
        'q': split according to generalized coordinates (len(qpos))
        'v': split according to degrees of freedom (len(qvel))
        'u': split according to actuators
        'a': split according to actuator activations
        'c': split according to camera
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      group_by: the type to group by, either joints, actuators, or cameras

    Returns:
      The stacked outputs of ``f`` matching the model's order.

    Raises:
        IndexError: if function output shape does not match out_types shape
    """
    _check_input(m, args, in_types)

    cache = _get_flat_cache(m, in_types, out_types, group_by)
    key_typ_ids_torch = cache["key_typ_ids_torch"]
    group_has_output = cache["group_has_output"]
    flat_ = cache["flat_"]

    # Use pre-extracted static args if available (pre-computed at device_put
    # time for known signatures), otherwise fall back to runtime extraction.
    all_static_args = cache.get("pre_extracted_static")
    if all_static_args is not None:
        # Derive static mask from the pre-extracted data (compile-safe).
        sample = next((g for g in all_static_args if g is not None), None)
        is_static = [x is not None for x in sample] if sample else [False] * len(args)
    else:
        is_static = _static_arg_mask(m, args)
        all_static_args = _extract_static_for_flat(args, in_types, key_typ_ids_torch, group_has_output, is_static)

    # Call vmap per group (Dynamo unrolls this constant-length loop)
    ys = []
    for i, ((_, typ_ids_t), has_output) in enumerate(zip(key_typ_ids_torch, group_has_output)):
        if has_output:
            f_args = [
                _take(arg, typ_ids_t[typ]) if not static else None
                for arg, typ, static in zip(args, in_types, is_static)
            ]
            y = _nvmap(f, all_static_args[i], *f_args)
            ys.append(y)
        else:
            ys.append(None)

    # Filter None results and matching cache entries
    active_kti = [v for y, v in zip(ys, key_typ_ids_torch) if y is not None]
    ys = [y for y in ys if y is not None]

    # Flatten grouped dimensions and concatenate across groups
    f_ret_is_seq = isinstance(ys[0], (list, tuple))
    ys = ys if f_ret_is_seq else [[y] for y in ys]
    ys = [[v if typ in flat_ else torch.flatten(v, 0, 1) for v, typ in zip(y, out_types)] for y in ys]
    ys = tree_map(lambda *x: concatenate(x), *ys)

    # Put concatenated results back in model order.
    n_expected = sum(cache["group_has_output"])
    assert len(active_kti) == n_expected, (
        f"scan.flat: {len(active_kti)} groups produced output but {n_expected} expected. "
        "All has_output groups must return non-None values."
    )
    reorder_indices = cache["reorder_indices"]
    reordered_ys = []
    for i, typ in enumerate(out_types):
        reordered_ys.append(_take(ys[i], reorder_indices[typ]))

    return reordered_ys if f_ret_is_seq else reordered_ys[0]


def _build_body_tree_cache(m, in_types, out_types, reverse):
    """Precompute grouping indices for scan.body_tree (numpy-heavy, runs once)."""
    jnt_type_np = _as_numpy(m.jnt_type)
    depths = np.zeros(m.nbody, dtype=np.int32)

    key_body_ids = {}
    for body_id in range(m.nbody):
        parent_id = -1
        if body_id > 0:
            parent_id = m.body_parentid[body_id]
        depths[body_id] = 1 + depths[parent_id]

        key = (depths[body_id],)
        for i, t in enumerate(out_types + in_types):
            id_ = parent_id if i < len(out_types) else body_id
            if t == "b":
                continue
            elif t == "j":
                key += tuple(jnt_type_np[np.nonzero(m.jnt_bodyid == id_)[0]])
            elif t == "v":
                key += (len(np.nonzero(m.dof_bodyid == id_)[0]),)
            elif t == "q":
                key += (len(np.nonzero(_q_bodyid(m) == id_)[0]),)

        body_ids = key_body_ids.get(key, np.array([], dtype=np.int32))
        key_body_ids[key] = np.append(body_ids, body_id)

    key_parents = {}
    for key, body_ids in key_body_ids.items():
        body_ids_nz = body_ids[body_ids != 0]
        if body_ids_nz.size == 0:
            continue
        pids = m.body_parentid[body_ids_nz]
        parents = {k for k, v in key_body_ids.items() if np.isin(v, pids).any()}
        key_parents[key] = list(sorted(parents))

    key_in_take, key_y_take = {}, {}
    for key, body_ids in key_body_ids.items():
        for i, typ in enumerate(in_types + out_types):
            if typ == "b":
                ids = body_ids
            elif typ == "j":
                ids = np.stack([np.nonzero(m.jnt_bodyid == b)[0] for b in body_ids])
            elif typ == "v":
                ids = np.stack([np.nonzero(m.dof_bodyid == b)[0] for b in body_ids])
            elif typ == "q":
                ids = np.stack([np.nonzero(_q_bodyid(m) == b)[0] for b in body_ids])
            else:
                raise ValueError(f"Unknown in_type: {typ}")
            if i < len(in_types):
                key_in_take.setdefault(key, []).append(ids)
            else:
                key_y_take.setdefault(key, []).append(np.hstack(ids))

    keys = sorted(key_body_ids, reverse=reverse)

    # Precompute carry-propagation index maps
    carry_maps = {}
    for key in keys:
        if reverse:
            child_keys = [k for k, v in key_parents.items() if key in v]
            child_info = []
            for child_key in child_keys:
                body_ids = key_body_ids[key]
                parent_ids = m.body_parentid[key_body_ids[child_key]]
                id_map = _index(body_ids, parent_ids)
                inv_idx, inv_mask = _invert_segment_ids(id_map, body_ids.size)
                child_info.append(
                    (
                        child_key,
                        _cached_long(inv_idx),
                        _DeviceCachedTensor(inv_mask),
                    )
                )
            carry_maps[key] = child_info
        elif key in key_parents:
            body_ids_all = [key_body_ids[p] for p in key_parents[key]]
            concat_body_ids = np.concatenate(body_ids_all)
            parent_ids = m.body_parentid[key_body_ids[key]]
            take_idx = _index(concat_body_ids, parent_ids)
            carry_maps[key] = (key_parents[key], _cached_long(take_idx))

    # Convert in_take indices to device-cached tensors
    key_in_take_torch = {}
    for key, take_list in key_in_take.items():
        key_in_take_torch[key] = [_cached_long(ids) for ids in take_list]

    # Pre-compute reorder indices for the compile path (assumes all keys
    # produce output, which is always true in production code).
    y_take_torch = []
    for i in range(len(out_types)):
        y_take_torch.append(_cached_long(np.argsort(np.concatenate([key_y_take[key][i] for key in keys]))))

    return {
        "key_in_take_torch": key_in_take_torch,
        "key_y_take": key_y_take,
        "keys": keys,
        "carry_maps": carry_maps,
        "y_take_torch": y_take_torch,
    }


def _get_body_tree_cache(m, in_types, out_types, reverse):
    """Get or build body_tree scan cache.

    For known signatures (registered in ``_KNOWN_BODY_TREE_CALLS``), the
    cache is pre-built at ``device_put`` time so this is a simple dict
    lookup with no graph break.
    """
    cache_key = (_model_cache_id(m), in_types, out_types, reverse)
    cache = _body_tree_cache.get(cache_key)
    if cache is None:
        cache = _build_body_tree_cache(m, in_types, out_types, reverse)
        _body_tree_cache[cache_key] = cache
    return cache


def body_tree(
    m: Model,
    f: Callable[..., Y],
    in_types: str,
    out_types: str,
    *args,
    reverse: bool = False,
) -> Y:
    r"""Scan ``f`` across bodies in tree order, carrying results up/down the tree.

    This function groups bodies according to level and attached joints, then calls
    vmap(f) on them.  Grouping indices are precomputed and cached per model.

    Args:
      m: an mjx mjmodel
      f: a function to be scanned with the following type signature:
          def f(y, *args) -> y
        where
          ``y`` is the carry value and return value
          ``*args`` are input arguments with types matching ``in_types``
      in_types: string specifying the type of each input arg:
        'b': split according to bodies
        'j': split according to joint types
        'q': split according to generalized coordinates (len(qpos))
        'v': split according to degrees of freedom (len(qvel))
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      reverse: if True, scans up the body tree from leaves to root, otherwise
        root to leaves

    Returns:
      The stacked outputs of ``f`` matching the model's body order.

    Raises:
        IndexError: if function output shape does not match out_types shape
    """
    _check_input(m, args, in_types)

    cache = _get_body_tree_cache(m, in_types, out_types, reverse)
    key_in_take_torch = cache["key_in_take_torch"]
    keys = cache["keys"]
    carry_maps = cache["carry_maps"]

    # Use pre-extracted static args if available (pre-computed at device_put
    # time for known signatures), otherwise fall back to runtime extraction.
    all_static_args = cache.get("pre_extracted_static")
    if all_static_args is not None:
        # Derive static mask from the pre-extracted data (compile-safe).
        # Skip carry position (index 0) which is always None.
        sample = next(iter(all_static_args.values()), None)
        is_static = [x is not None for x in sample[1:]] if sample else [False] * len(args)
    else:
        is_static = _static_arg_mask(m, args)
        all_static_args = _extract_static_for_body_tree(args, key_in_take_torch, keys, is_static)

    # Scan over groups in tree order, carrying results up/down
    key_y = {}
    for key in keys:
        carry = None

        if reverse:
            for child_key, inv_idx, inv_mask in carry_maps[key]:
                y = key_y[child_key]
                if y is None:
                    continue

                def gather_sum(x, ii=inv_idx, im=inv_mask):
                    return _gather_segment_sum(x, ii, im)

                y = tree_map(gather_sum, y)
                carry = y if carry is None else tree_map(torch.add, carry, y)
        elif key in carry_maps:
            parent_keys, take_idx = carry_maps[key]
            ys_all = [key_y[p] for p in parent_keys]
            ys_filtered = [y_val for y_val in ys_all if y_val is not None]
            if ys_filtered:
                y = tree_map(lambda *x: concatenate(x), *ys_filtered) if len(ys_filtered) > 1 else ys_filtered[0]
                take_fn = lambda x, i=take_idx: _take(x, i)
                carry = tree_map(take_fn, y)

        ids_list = key_in_take_torch[key]
        f_args = [_take(arg, ids) if not static else None for arg, ids, static in zip(args, ids_list, is_static)]
        y = _nvmap(f, all_static_args[key], carry, *f_args)

        key_y[key] = y

    # Filter out keys whose callback returned None
    active_keys = [k for k in keys if key_y[k] is not None]

    # Concatenate results, drop grouping dimensions, put back in model order
    assert len(active_keys) == len(keys), (
        f"scan.body_tree: {len(active_keys)} groups produced output but {len(keys)} expected. "
        "All groups must return non-None values."
    )
    y = []
    for i, typ in enumerate(out_types):
        y_typ = [key_y[key] for key in active_keys]
        if len(out_types) > 1:
            y_typ = [y_[i] for y_ in y_typ]
        if typ != "b":
            y_typ = tree_map(lambda x: torch.flatten(x, 0, 1), y_typ)
        y_typ = tree_map(lambda *x: _cat_device_safe(x), *y_typ)
        y.append(_take(y_typ, cache["y_take_torch"][i]))

    y = y[0] if len(out_types) == 1 else y

    return y


def segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Computes the sum within segments of a tensor.

    Equivalent to jax.ops.segment_sum.

    Args:
      data: values to segment-sum
      segment_ids: segment index for each element in data
      num_segments: total number of segments

    Returns:
      Tensor of shape (num_segments, *data.shape[1:]) with summed values.
    """
    if isinstance(segment_ids, _DeviceCachedTensor):
        segment_ids = segment_ids.to(data.device).long()
    elif not isinstance(segment_ids, torch.Tensor):
        segment_ids = torch.tensor(segment_ids, device=data.device, dtype=torch.long)
    else:
        segment_ids = segment_ids.to(device=data.device, dtype=torch.long)

    data = data.clone().contiguous()
    shape = (num_segments,) + data.shape[1:]
    result = torch.zeros(shape, dtype=data.dtype, device=data.device)
    # Use out-of-place operations to be compatible with vmap
    idx = segment_ids
    for _ in range(data.ndim - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(data).clone()
    result = result.scatter_add(0, idx, data)
    return result
