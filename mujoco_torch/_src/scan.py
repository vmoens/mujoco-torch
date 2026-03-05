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
"""Scan across data ordered by body joint types and kinematic tree order.

Single-vmap implementation: ``flat()`` issues one ``torch.vmap`` call per
invocation; ``body_tree()`` issues one per depth level.  Type-dependent
logic is handled inside callbacks via ``torch.where``.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import torch
from tensordict import UnbatchedTensor
from torch.utils._pytree import tree_map

from mujoco_torch._src.types import JointType, Model, TrnType

Y = TypeVar("Y")

MAX_QPOS_PER_JNT = 7  # FREE joint has the most: 7 qpos values
MAX_DOF_PER_JNT = 6  # FREE joint has the most: 6 dofs

_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}  # JointType int -> qpos width
_DOF_WIDTH = {0: 6, 1: 3, 2: 1, 3: 1}  # JointType int -> dof width

# Module-level caches.  Keyed by (cache_id, in_types, out_types, group_by_or_reverse).
_flat_cache: dict = {}
_body_tree_cache: dict = {}


def _model_cache_id(m: Model) -> int:
    return m.cache_id


def clear_scan_caches():
    """Clear precomputed scan grouping caches (useful for testing)."""
    _flat_cache.clear()
    _body_tree_cache.clear()


def _resolve_cached_tensors(obj, device: torch.device):
    """Recursively replace _DeviceCachedTensor and CPU tensors with device tensors."""
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
    """Resolve all _DeviceCachedTensor entries for *cache_id* to *device*."""
    device = torch.device(device)
    for key in list(_flat_cache):
        if key[0] == cache_id:
            _flat_cache[key] = _resolve_cached_tensors(_flat_cache[key], device)
    for key in list(_body_tree_cache):
        if key[0] == cache_id:
            _body_tree_cache[key] = _resolve_cached_tensors(_body_tree_cache[key], device)


# Known scan call-site signatures: (in_types, out_types, group_by_or_reverse).
_KNOWN_FLAT_CALLS = [
    ("jbbjj", "v", "j"),
    ("uuua", "a", "u"),
    ("uuuuuuuuu", "u", "u"),
    ("uuaau", "a", "u"),
    ("au", "a", "u"),
    ("jqv", "q", "j"),
    ("jjqq", "v", "j"),
]

_KNOWN_BODY_TREE_CALLS = [
    ("jjjqqbbb", "qjjbbb", False),
    ("bb", "bb", True),
    ("b", "b", True),
    ("jvvb", "bv", False),
    ("vvvvjb", "b", False),
]


def _signature_applies(m, in_types, out_types, group_by=None):
    all_types = set(in_types + out_types)
    if group_by is not None:
        all_types.add(group_by)
    if ("u" in all_types or "a" in all_types) and m.nu == 0:
        return False
    if ("j" in all_types or "q" in all_types or "v" in all_types) and m.njnt == 0:
        return False
    return True


def precompute_scan_caches(m, cache_id: int):
    """Pre-build all scan caches for known call-site signatures."""
    jnt_type_np = np.array(m.jnt_type)
    jt_set = frozenset(int(t) for t in jnt_type_np) if m.njnt > 0 else frozenset()
    max_qw = max((_QPOS_WIDTH[t] for t in jt_set), default=1)
    max_dw = max((_DOF_WIDTH[t] for t in jt_set), default=1)

    for in_types, out_types, group_by in _KNOWN_FLAT_CALLS:
        if not _signature_applies(m, in_types, out_types, group_by):
            continue
        _flat_cache[(cache_id, in_types, out_types, group_by)] = _build_flat_cache(
            m, in_types, out_types, group_by, max_qw, max_dw,
        )

    for in_types, out_types, reverse in _KNOWN_BODY_TREE_CALLS:
        if not _signature_applies(m, in_types, out_types):
            continue
        cache = _build_body_tree_cache(m, in_types, out_types, reverse, max_qw, max_dw)
        _body_tree_cache[(cache_id, in_types, out_types, reverse)] = cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np_to_long(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.long)


class _DeviceCachedTensor:
    """A CPU index tensor that lazily caches per-device copies."""

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
    cpu_tensor = torch.as_tensor(np.asarray(x), dtype=torch.long).cpu()
    return _DeviceCachedTensor(cpu_tensor)


def _cat_device_safe(tensors):
    device = None
    for t in tensors:
        if t.device.type != "cpu":
            device = t.device
            break
    if device is not None:
        tensors = tuple(t.to(device) if t.device != device else t for t in tensors)
    return torch.cat(tensors)


def _take(obj: Y, idx) -> Y:
    """Takes idxs on any pytree given to it."""
    if isinstance(obj, UnbatchedTensor):
        obj = obj.data
    if isinstance(obj, np.ndarray):
        if isinstance(idx, _DeviceCachedTensor):
            return obj[idx._cpu.numpy()]
        if isinstance(idx, torch.Tensor):
            return obj[idx.cpu().numpy()]
        return obj[idx]

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
    if isinstance(val, UnbatchedTensor):
        return val.data.cpu().numpy()
    if isinstance(val, torch.Tensor):
        return val.cpu().numpy()
    return val


def _q_bodyid(m: Model) -> np.ndarray:
    """Returns the bodyid for each qpos address (Model version)."""
    jnt_type = _as_numpy(m.jnt_type)
    q_bodyids = [np.array([], dtype=np.int32)]
    for jt, jnt_bodyid in zip(jnt_type, m.jnt_bodyid):
        width = {JointType.FREE: 7, JointType.BALL: 4}.get(jt, 1)
        q_bodyids.append(np.repeat(jnt_bodyid, width))
    return np.concatenate(q_bodyids)


def _q_bodyid_np(m) -> np.ndarray:
    """Returns the bodyid for each qpos address (raw MjModel version)."""
    jnt_type = np.array(m.jnt_type)
    q_bodyids = [np.array([], dtype=np.int32)]
    for jt, jnt_bodyid in zip(jnt_type, m.jnt_bodyid):
        width = {0: 7, 1: 4}.get(int(jt), 1)
        q_bodyids.append(np.repeat(jnt_bodyid, width))
    return np.concatenate(q_bodyids)


def _q_jointid(m: Model) -> np.ndarray:
    jnt_type = _as_numpy(m.jnt_type)
    q_jointid = [np.array([], dtype=np.int32)]
    for i, jt in enumerate(jnt_type):
        width = {JointType.FREE: 7, JointType.BALL: 4}.get(jt, 1)
        q_jointid.append(np.repeat(i, width))
    return np.concatenate(q_jointid)


def _index(haystack, needle):
    if isinstance(haystack, torch.Tensor):
        idx = torch.argsort(haystack)
        sorted_haystack = haystack[idx]
        sorted_idx = torch.searchsorted(sorted_haystack, needle)
        sorted_idx = sorted_idx.clamp(max=idx.shape[0] - 1)
        result = idx[sorted_idx]
        result = torch.where(haystack[result] == needle, result, torch.full_like(result, -1))
        return result
    idx = np.argsort(haystack)
    sorted_haystack = haystack[idx]
    sorted_idx = np.searchsorted(sorted_haystack, needle)
    idx = np.take(idx, sorted_idx, mode="clip")
    idx[haystack[idx] != needle] = -1
    return idx


def _pad_1d(arr, max_len):
    """Pad a 1D numpy array to max_len using edge padding."""
    n = len(arr)
    if n == 0:
        return np.zeros(max_len, dtype=arr.dtype)
    if n >= max_len:
        return arr[:max_len]
    return np.pad(arr, (0, max_len - n), mode="edge")


def _nvmap(f: Callable[..., Y], *args) -> Y:
    """A vmap wrapper that handles None arguments."""

    def _check_empty(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor) and a.shape[0] == 0:
            return None
        return a

    args = [_check_empty(a) for a in args]
    none_mask = [a is None for a in args]
    vmap_args = [a for a in args if a is not None]

    def _ensure_tensor(x):
        if isinstance(x, (np.ndarray, np.integer, np.floating)):
            return torch.as_tensor(x)
        if isinstance(x, (int, float, bool)):
            return torch.as_tensor(x)
        return x

    def outer_f(*vmap_inputs, _none_mask=none_mask):
        it = iter(vmap_inputs)
        full_args = [None if m else next(it) for m in _none_mask]
        result = f(*full_args)
        return tree_map(_ensure_tensor, result)

    if not vmap_args:
        return f(*[None] * len(args))

    if torch.compiler.is_compiling():
        return torch.vmap(outer_f, in_dims=tuple([0] * len(vmap_args)))(*vmap_args)

    try:
        return torch.vmap(outer_f, in_dims=tuple([0] * len(vmap_args)))(*vmap_args)
    except ValueError as e:
        if "NoneType" in str(e):
            return None
        raise


def _check_input(m: Model, args: Any, in_types: str) -> None:
    if torch.compiler.is_compiling():
        return
    size = {
        "b": m.nbody, "j": m.njnt, "q": m.nq, "v": m.nv,
        "u": m.nu, "a": m.na, "s": m.nsite, "c": m.ncam,
    }
    for idx, (arg, typ) in enumerate(zip(args, in_types)):
        arg_len = arg.data.shape[0] if isinstance(arg, UnbatchedTensor) else len(arg)
        if arg_len != size[typ]:
            raise IndexError(
                f'f argument "{idx}" with type "{typ}" has length "{arg_len}"'
                f" which does not match the in_types[{idx}] expected length of "
                f'"{size[typ]}".'
            )


# ---------------------------------------------------------------------------
# Per-body index helpers for group_by="j"
#
# For types "q" and "v", indices are organized per-joint:
#   "q": (nbody, max_j, MAX_QPOS_PER_JNT)  -- 7 qpos slots per joint
#   "v": (nbody, max_j, MAX_DOF_PER_JNT)   -- 6 dof  slots per joint
#   "j": (nbody, max_j)                     -- 1 joint index per joint
#   "b": (nbody,)                           -- 1 body index
#
# After _take, the callback receives per-joint arrays.  The scan function
# flattens the per-joint dims in the output before gathering.
# ---------------------------------------------------------------------------

def _build_body_joint_indices(m, all_types, max_j, max_qw, max_dw):
    """Compute per-body padded index tensors for group_by='j'."""
    jnt_type_np = _as_numpy(m.jnt_type)
    nbody = m.nbody

    padded: dict[str, Any] = {}

    if "b" in all_types:
        padded["b"] = _cached_long(np.arange(nbody, dtype=np.int64))

    if "j" in all_types:
        j_arr = np.zeros((nbody, max_j), dtype=np.int64)
        for i in range(nbody):
            jids = np.nonzero(m.jnt_bodyid == i)[0]
            for k, jid in enumerate(jids):
                j_arr[i, k] = jid
            if len(jids) > 0:
                j_arr[i, len(jids):] = jids[-1]
        padded["j"] = _cached_long(j_arr)

    if "q" in all_types:
        q_arr = np.zeros((nbody, max_j, max_qw), dtype=np.int64)
        jnt_qposadr = np.array(m.jnt_qposadr, dtype=np.int64)
        for i in range(nbody):
            jids = np.nonzero(m.jnt_bodyid == i)[0]
            for k, jid in enumerate(jids):
                jt = int(jnt_type_np[jid])
                qw = _QPOS_WIDTH[jt]
                qa = jnt_qposadr[jid]
                q_ids = np.arange(qa, qa + qw, dtype=np.int64)
                q_arr[i, k, :qw] = q_ids
                if qw > 0:
                    q_arr[i, k, qw:] = q_ids[-1]
            if len(jids) > 0:
                q_arr[i, len(jids):] = q_arr[i, len(jids) - 1]
        padded["q"] = _cached_long(q_arr)

    if "v" in all_types:
        v_arr = np.zeros((nbody, max_j, max_dw), dtype=np.int64)
        jnt_dofadr = np.array(m.jnt_dofadr, dtype=np.int64)
        for i in range(nbody):
            jids = np.nonzero(m.jnt_bodyid == i)[0]
            for k, jid in enumerate(jids):
                jt = int(jnt_type_np[jid])
                dw = _DOF_WIDTH[jt]
                da = jnt_dofadr[jid]
                d_ids = np.arange(da, da + dw, dtype=np.int64)
                v_arr[i, k, :dw] = d_ids
                if dw > 0:
                    v_arr[i, k, dw:] = d_ids[-1]
            if len(jids) > 0:
                v_arr[i, len(jids):] = v_arr[i, len(jids) - 1]
        padded["v"] = _cached_long(v_arr)

    return padded


def _build_gather_indices_j(m, out_types, max_j, max_qw, max_dw):
    """Compute output gather indices for group_by='j'.

    Returns dict mapping output type to (body_idx, offset_idx) or None.
    Also returns per_joint_inner_width mapping type -> inner width for
    flattening per-joint output dimensions.
    """
    gather: dict[str, tuple | None] = {}
    inner_w: dict[str, int | None] = {}

    for t in set(out_types):
        if t == "b":
            gather[t] = None
            inner_w[t] = None
        elif t == "j":
            body_list, off_list = [], []
            for jid in range(m.njnt):
                bid = int(m.jnt_bodyid[jid])
                local_j = int(np.count_nonzero(m.jnt_bodyid[:jid] == bid))
                body_list.append(bid)
                off_list.append(local_j)
            gather[t] = (
                _cached_long(np.array(body_list, dtype=np.int64)),
                _cached_long(np.array(off_list, dtype=np.int64)),
            )
            inner_w[t] = None
        elif t == "q":
            body_list, off_list = [], []
            jnt_qposadr = np.array(m.jnt_qposadr, dtype=np.int64)
            q_bid = _q_bodyid(m)
            q_jid = _q_jointid(m)
            for k in range(m.nq):
                bid = int(q_bid[k])
                jid = int(q_jid[k])
                local_j = int(np.count_nonzero(m.jnt_bodyid[:jid] == bid))
                qpos_in_jnt = k - int(jnt_qposadr[jid])
                body_list.append(bid)
                off_list.append(local_j * max_qw + qpos_in_jnt)
            gather[t] = (
                _cached_long(np.array(body_list, dtype=np.int64)),
                _cached_long(np.array(off_list, dtype=np.int64)),
            )
            inner_w[t] = max_qw
        elif t == "v":
            body_list, off_list = [], []
            jnt_dofadr = np.array(m.jnt_dofadr, dtype=np.int64)
            dof_jntid = np.array(m.dof_jntid, dtype=np.int64)
            for k in range(m.nv):
                bid = int(m.dof_bodyid[k])
                jid = int(dof_jntid[k])
                local_j = int(np.count_nonzero(m.jnt_bodyid[:jid] == bid))
                dof_in_jnt = k - int(jnt_dofadr[jid])
                body_list.append(bid)
                off_list.append(local_j * max_dw + dof_in_jnt)
            gather[t] = (
                _cached_long(np.array(body_list, dtype=np.int64)),
                _cached_long(np.array(off_list, dtype=np.int64)),
            )
            inner_w[t] = max_dw

    return gather, inner_w


# ---------------------------------------------------------------------------
# Flat scan (single vmap)
# ---------------------------------------------------------------------------

def _build_flat_cache(m, in_types, out_types, group_by, max_qw=MAX_QPOS_PER_JNT, max_dw=MAX_DOF_PER_JNT):
    all_types = set(in_types + out_types)
    flat_types = {"j": "b", "u": "uaj", "c": "c"}[group_by]

    if group_by == "j":
        max_j = max(1, int(np.max([np.count_nonzero(m.jnt_bodyid == i) for i in range(m.nbody)]))) if m.nbody else 1
        padded_indices = _build_body_joint_indices(m, all_types, max_j, max_qw, max_dw)
        gather_indices, per_joint_inner_w = _build_gather_indices_j(m, out_types, max_j, max_qw, max_dw)
        return {
            "padded_indices": padded_indices,
            "gather_indices": gather_indices,
            "per_joint_inner_w": per_joint_inner_w,
            "flat_types": flat_types,
            "n_items": m.nbody,
        }

    elif group_by == "u":
        n_items = m.nu
        padded_indices: dict[str, Any] = {}
        for t in all_types:
            if t == "u":
                padded_indices[t] = _cached_long(np.arange(n_items, dtype=np.int64))
            elif t == "a":
                actadr = np.array(m.actuator_actadr, dtype=np.int64)
                padded_indices[t] = _cached_long(np.where(actadr < 0, 0, actadr))
            elif t == "j":
                jnt_ids = np.array(m.actuator_trnid[:, 0], dtype=np.int64)
                padded_indices[t] = _cached_long(np.where(jnt_ids < 0, 0, jnt_ids))
            elif t == "v":
                v_list = []
                for i in range(n_items):
                    if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT):
                        jid = m.actuator_trnid[i, 0]
                        v_list.append(np.nonzero(m.dof_jntid == jid)[0].astype(np.int64))
                    else:
                        v_list.append(np.array([0], dtype=np.int64))
                max_v = max(len(a) for a in v_list) if v_list else 1
                padded_indices[t] = _cached_long(np.stack([_pad_1d(a, max_v) for a in v_list]))
            elif t == "q":
                q_list = []
                for i in range(n_items):
                    if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT):
                        jid = m.actuator_trnid[i, 0]
                        q_list.append(np.nonzero(_q_jointid(m) == jid)[0].astype(np.int64))
                    else:
                        q_list.append(np.array([0], dtype=np.int64))
                max_q = max(len(a) for a in q_list) if q_list else 1
                padded_indices[t] = _cached_long(np.stack([_pad_1d(a, max_q) for a in q_list]))
            elif t == "s":
                site_ids = np.array(m.actuator_trnid, dtype=np.int64)
                padded_indices[t] = _cached_long(np.where(site_ids < 0, 0, site_ids))

        gather_indices: dict[str, tuple | None] = {}
        for t in set(out_types):
            if t in flat_types:
                gather_indices[t] = None
            elif t == "a":
                actadr = np.array(m.actuator_actadr, dtype=np.int64)
                valid = np.nonzero(actadr >= 0)[0]
                gather_indices[t] = (_cached_long(valid[np.argsort(actadr[valid])]),)
            else:
                gather_indices[t] = None

        return {
            "padded_indices": padded_indices,
            "gather_indices": gather_indices,
            "per_joint_inner_w": {},
            "flat_types": flat_types,
            "n_items": n_items,
        }

    elif group_by == "c":
        padded_indices = {}
        for t in all_types:
            if t == "c":
                padded_indices[t] = _cached_long(np.arange(m.ncam, dtype=np.int64))
        return {
            "padded_indices": padded_indices,
            "gather_indices": {t: None for t in set(out_types)},
            "per_joint_inner_w": {},
            "flat_types": "c",
            "n_items": m.ncam,
        }
    else:
        raise NotImplementedError(f'group by type "{group_by}" not implemented.')


def _get_flat_cache(m, in_types, out_types, group_by):
    cache_key = (_model_cache_id(m), in_types, out_types, group_by)
    cache = _flat_cache.get(cache_key)
    if cache is None:
        cache = _build_flat_cache(m, in_types, out_types, group_by, m.max_qpos_per_jnt, m.max_dof_per_jnt)
        _flat_cache[cache_key] = cache
    return cache


def _gather_output(yi, gi, inner_w):
    """Gather a single output tensor using precomputed indices."""
    if gi is None:
        return yi
    if len(gi) == 1:
        idx = gi[0]
        idx = idx.to(yi.device) if isinstance(idx, _DeviceCachedTensor) else idx
        return yi[idx]

    body_idx, offset_idx = gi
    bi = body_idx.to(yi.device) if isinstance(body_idx, _DeviceCachedTensor) else body_idx
    oi = offset_idx.to(yi.device) if isinstance(offset_idx, _DeviceCachedTensor) else offset_idx

    # Flatten per-joint inner dimension if needed:
    #   (batch, max_j, inner_w, ...) -> (batch, max_j * inner_w, ...)
    if inner_w is not None and yi.ndim >= 3:
        yi = yi.reshape(yi.shape[0], -1, *yi.shape[3:])

    return yi[bi, oi]


def flat(
    m: Model,
    f: Callable[..., Y],
    in_types: str,
    out_types: str,
    *args,
    group_by: str = "j",
) -> Y:
    r"""Scan a function across bodies or actuators using a single vmap call.

    All items are processed in a single ``torch.vmap`` call. Type-dependent
    logic is handled inside the callback via ``torch.where``.  For
    ``group_by="j"``, per-joint inputs (types ``"q"`` and ``"v"``) are
    organized as ``(max_j, 7)`` and ``(max_j, 6)`` respectively inside the
    vmapped function, enabling fixed-offset indexing with no data-dependent
    slicing.

    Args:
      m: an mjx model
      f: a function to be scanned
      in_types: string specifying the type of each input arg
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      group_by: the type to group by ("j", "u", or "c")

    Returns:
      The outputs of ``f`` in model order.
    """
    _check_input(m, args, in_types)

    cache = _get_flat_cache(m, in_types, out_types, group_by)
    padded = cache["padded_indices"]
    gather = cache["gather_indices"]
    inner_w = cache["per_joint_inner_w"]

    f_args = [_take(arg, padded[typ]) for arg, typ in zip(args, in_types)]
    y = _nvmap(f, *f_args)

    if y is None:
        return None

    is_seq = isinstance(y, (list, tuple))
    ys = list(y) if is_seq else [y]
    result = [_gather_output(yi, gather.get(typ), inner_w.get(typ)) for yi, typ in zip(ys, out_types)]

    return tuple(result) if is_seq else result[0]


# ---------------------------------------------------------------------------
# Body-tree scan (per-depth single vmap)
# ---------------------------------------------------------------------------

def _build_body_tree_cache(m, in_types, out_types, reverse, max_qw=MAX_QPOS_PER_JNT, max_dw=MAX_DOF_PER_JNT):
    jnt_type_np = _as_numpy(m.jnt_type)

    # Compute depths
    depths = np.zeros(m.nbody, dtype=np.int32)
    for body_id in range(1, m.nbody):
        depths[body_id] = 1 + depths[m.body_parentid[body_id]]

    max_depth = int(depths.max()) if m.nbody > 0 else 0
    max_j = max(1, int(np.max([np.count_nonzero(m.jnt_bodyid == i) for i in range(m.nbody)]))) if m.nbody else 1

    # Bodies per depth level
    depth_body_ids: dict[int, np.ndarray] = {}
    for d in range(max_depth + 1):
        depth_body_ids[d] = np.nonzero(depths == d)[0].astype(np.int64)

    # Per-depth padded indices (re-use body-joint helpers restricted to each depth's bodies)
    per_depth_padded: dict[int, dict[str, Any]] = {}
    for d in range(max_depth + 1):
        bids = depth_body_ids[d]
        pd: dict[str, Any] = {}

        if "b" in set(in_types):
            pd["b"] = _cached_long(bids)

        if "j" in set(in_types):
            j_arr = np.zeros((len(bids), max_j), dtype=np.int64)
            for li, b in enumerate(bids):
                jids = np.nonzero(m.jnt_bodyid == b)[0]
                for k, jid in enumerate(jids):
                    j_arr[li, k] = jid
                if len(jids) > 0:
                    j_arr[li, len(jids):] = jids[-1]
            pd["j"] = _cached_long(j_arr)

        if "q" in set(in_types):
            jnt_qposadr = np.array(m.jnt_qposadr, dtype=np.int64)
            q_arr = np.zeros((len(bids), max_j, max_qw), dtype=np.int64)
            for li, b in enumerate(bids):
                jids = np.nonzero(m.jnt_bodyid == b)[0]
                for k, jid in enumerate(jids):
                    jt = int(jnt_type_np[jid])
                    qw = _QPOS_WIDTH[jt]
                    qa = jnt_qposadr[jid]
                    q_ids = np.arange(qa, qa + qw, dtype=np.int64)
                    q_arr[li, k, :qw] = q_ids
                    if qw > 0:
                        q_arr[li, k, qw:] = q_ids[-1]
                if len(jids) > 0:
                    q_arr[li, len(jids):] = q_arr[li, len(jids) - 1]
            pd["q"] = _cached_long(q_arr)

        if "v" in set(in_types):
            jnt_dofadr = np.array(m.jnt_dofadr, dtype=np.int64)
            v_arr = np.zeros((len(bids), max_j, max_dw), dtype=np.int64)
            for li, b in enumerate(bids):
                jids = np.nonzero(m.jnt_bodyid == b)[0]
                for k, jid in enumerate(jids):
                    jt = int(jnt_type_np[jid])
                    dw = _DOF_WIDTH[jt]
                    da = jnt_dofadr[jid]
                    d_ids = np.arange(da, da + dw, dtype=np.int64)
                    v_arr[li, k, :dw] = d_ids
                    if dw > 0:
                        v_arr[li, k, dw:] = d_ids[-1]
                if len(jids) > 0:
                    v_arr[li, len(jids):] = v_arr[li, len(jids) - 1]
            pd["v"] = _cached_long(v_arr)

        per_depth_padded[d] = pd

    # Carry propagation
    carry_maps: dict[int, Any] = {}
    if not reverse:
        for d in range(1, max_depth + 1):
            bids = depth_body_ids[d]
            parent_bids = np.array([m.body_parentid[b] for b in bids], dtype=np.int64)
            prev_bids = depth_body_ids[d - 1]
            carry_maps[d] = _cached_long(_index(prev_bids, parent_bids))
    else:
        for d in range(max_depth - 1, -1, -1):
            if d + 1 > max_depth:
                carry_maps[d] = None
                continue
            child_bids = depth_body_ids[d + 1]
            parent_bids_child = np.array([m.body_parentid[b] for b in child_bids], dtype=np.int64)
            cur_bids = depth_body_ids[d]
            carry_maps[d] = (_cached_long(_index(cur_bids, parent_bids_child)), len(cur_bids))

    # Output gather
    depth_order = list(range(max_depth, -1, -1)) if reverse else list(range(max_depth + 1))
    concat_body_order = np.concatenate([depth_body_ids[d] for d in depth_order])

    # Build gather similar to flat scan
    gather_indices, per_joint_inner_w = _build_gather_indices_j(m, out_types, max_j, max_qw, max_dw)

    # For body_tree, we need to map from concat body order to model order
    body_to_concat_pos = np.empty(m.nbody, dtype=np.int64)
    cp = 0
    for d in depth_order:
        bids = depth_body_ids[d]
        for li, b in enumerate(bids):
            body_to_concat_pos[b] = cp + li
        cp += len(bids)

    # Remap gather indices to use concat position instead of body id
    output_gather: dict[str, tuple | None] = {}
    output_inner_w: dict[str, int | None] = {}
    for t in set(out_types):
        gi = gather_indices.get(t)
        iw = per_joint_inner_w.get(t)
        output_inner_w[t] = iw
        if gi is None:
            # "b" type: simple reorder
            output_gather[t] = (_cached_long(np.argsort(concat_body_order)),)
        else:
            body_idx_np = gi[0]._cpu.numpy()
            offset_idx_np = gi[1]._cpu.numpy()
            # Remap body ids to concat positions
            remapped_body = body_to_concat_pos[body_idx_np]
            output_gather[t] = (
                _cached_long(remapped_body),
                _cached_long(offset_idx_np),
            )

    return {
        "depth_body_ids": depth_body_ids,
        "per_depth_padded": per_depth_padded,
        "carry_maps": carry_maps,
        "output_gather": output_gather,
        "output_inner_w": output_inner_w,
        "depth_order": depth_order,
        "max_depth": max_depth,
    }


def _get_body_tree_cache(m, in_types, out_types, reverse):
    cache_key = (_model_cache_id(m), in_types, out_types, reverse)
    cache = _body_tree_cache.get(cache_key)
    if cache is None:
        cache = _build_body_tree_cache(m, in_types, out_types, reverse, m.max_qpos_per_jnt, m.max_dof_per_jnt)
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
    r"""Scan ``f`` across bodies in tree order with a single vmap per depth level.

    Args:
      m: an mjx model
      f: a function ``f(carry, *args) -> y``
      in_types: string specifying the type of each input arg
      out_types: string specifying the types the output dimension matches
      *args: the input arguments corresponding to ``in_types``
      reverse: if True, scan from leaves to root

    Returns:
      The stacked outputs of ``f`` matching the model's order.
    """
    _check_input(m, args, in_types)

    cache = _get_body_tree_cache(m, in_types, out_types, reverse)
    depth_order = cache["depth_order"]
    per_depth_padded = cache["per_depth_padded"]
    carry_maps = cache["carry_maps"]
    output_gather = cache["output_gather"]
    output_inner_w = cache["output_inner_w"]

    depth_y: dict[int, Any] = {}
    for d in depth_order:
        padded = per_depth_padded[d]

        carry = None
        if not reverse and d > 0:
            parent_index = carry_maps[d]
            prev_y = depth_y[d - 1]
            if prev_y is not None:
                carry = tree_map(lambda x, i=parent_index: _take(x, i), prev_y)
        elif reverse and d < cache["max_depth"]:
            cm = carry_maps[d]
            if cm is not None:
                seg_ids, n_seg = cm
                child_y = depth_y[d + 1]
                if child_y is not None:
                    carry = tree_map(lambda x, i=seg_ids, s=n_seg: segment_sum(x, i, s), child_y)

        f_args = [_take(arg, padded[typ]) for arg, typ in zip(args, in_types)]
        y = _nvmap(f, carry, *f_args)
        depth_y[d] = y

    # Concatenate across depths and gather into model order
    result = []
    for i, typ in enumerate(out_types):
        y_parts = []
        for d in depth_order:
            y_d = depth_y[d]
            if y_d is None:
                continue
            part = y_d[i] if len(out_types) > 1 else y_d
            y_parts.append(part)

        if not y_parts:
            result.append(torch.empty(0))
            continue

        y_cat = tree_map(lambda *x: _cat_device_safe(x), *y_parts) if len(y_parts) > 1 else y_parts[0]

        gi = output_gather[typ]
        iw = output_inner_w.get(typ)
        if gi is not None and len(gi) == 1:
            idx = gi[0]
            idx = idx.to(y_cat.device) if isinstance(idx, _DeviceCachedTensor) else idx
            result.append(y_cat[idx])
        elif gi is not None and len(gi) == 2:
            result.append(_gather_output(y_cat, gi, iw))
        else:
            result.append(y_cat)

    return result[0] if len(out_types) == 1 else result


def segment_sum(
    data: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Computes the sum within segments of a tensor."""
    if isinstance(segment_ids, _DeviceCachedTensor):
        segment_ids = segment_ids.to(data.device).long()
    elif not isinstance(segment_ids, torch.Tensor):
        segment_ids = torch.tensor(segment_ids, device=data.device, dtype=torch.long)
    else:
        segment_ids = segment_ids.to(device=data.device, dtype=torch.long)

    data = data.clone().contiguous()
    shape = (num_segments,) + data.shape[1:]
    result = torch.zeros(shape, dtype=data.dtype, device=data.device)
    idx = segment_ids
    for _ in range(data.ndim - 1):
        idx = idx.unsqueeze(-1)
    idx = idx.expand_as(data).clone()
    result = result.scatter_add(0, idx, data)
    return result
