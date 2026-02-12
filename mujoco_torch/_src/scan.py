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

import weakref
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from mujoco_torch._src.math import concatenate
from mujoco_torch._src.types import JointType
from mujoco_torch._src.types import Model
from mujoco_torch._src.types import TrnType
from torch.utils._pytree import tree_map

Y = TypeVar('Y')

# Module-level caches for scan grouping indices.  Keyed by
# (model_id, in_types, out_types, group_by_or_reverse).
# Using WeakValueDictionary is not possible here since the values are plain
# dicts/lists, so we use a regular dict with id-based keys.  Models are
# long-lived so this is fine.
_flat_cache: dict = {}
_body_tree_cache: dict = {}
_model_cache_id_counter = 0
_model_cache_ids: dict = {}  # id(m) -> cache_id (monotonic)


def _model_cache_id(m: Model) -> int:
  """Return a unique cache ID for this model instance.

  Uses object identity but avoids id() reuse issues by tracking a
  monotonic counter and validating the object is still the same.
  """
  global _model_cache_id_counter
  mid = id(m)
  # Check if we already assigned a cache_id to this exact object
  stored = _model_cache_ids.get(mid)
  if stored is not None:
    cache_id, ref = stored
    # Validate it's still the same object (not a recycled id)
    if ref is m:
      return cache_id
  # New model or recycled id - assign a new cache_id
  _model_cache_id_counter += 1
  cache_id = _model_cache_id_counter
  _model_cache_ids[mid] = (cache_id, m)
  return cache_id


def clear_scan_caches():
  """Clear precomputed scan grouping caches (useful for testing)."""
  _flat_cache.clear()
  _body_tree_cache.clear()
  _model_cache_ids.clear()


def _np_to_long(x):
  """Convert numpy array to torch.LongTensor."""
  return torch.as_tensor(np.asarray(x), dtype=torch.long)


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
    if isinstance(idx, torch.Tensor):
      return obj[idx.numpy()]
    return obj[idx]

  # Ensure idx is a torch tensor for torch pytree operations
  if not isinstance(idx, torch.Tensor):
    idx = _np_to_long(idx)

  def take(x):
    if not x.shape[0]:
      return x
    return x[idx]

  return tree_map(take, obj)


def _q_bodyid(m: Model) -> np.ndarray:
  """Returns the bodyid for each qpos adress."""
  q_bodyids = [np.array([], dtype=np.int32)]
  for jnt_type, jnt_bodyid in zip(m.jnt_type, m.jnt_bodyid):
    width = {JointType.FREE: 7, JointType.BALL: 4}.get(jnt_type, 1)
    q_bodyids.append(np.repeat(jnt_bodyid, width))
  return np.concatenate(q_bodyids)


def _q_jointid(m: Model) -> np.ndarray:
  """Returns the jointid for each qpos adress."""
  q_jointid = [np.array([], dtype=np.int32)]
  for i, jnt_type in enumerate(m.jnt_type):
    width = {JointType.FREE: 7, JointType.BALL: 4}.get(jnt_type, 1)
    q_jointid.append(np.repeat(i, width))
  return np.concatenate(q_jointid)


def _index(haystack, needle):
  """Returns indexes in haystack for elements in needle.

  Works with both numpy arrays and torch tensors.
  """
  if isinstance(haystack, torch.Tensor):
    idx = torch.argsort(haystack)
    sorted_haystack = haystack[idx]
    sorted_idx = torch.searchsorted(sorted_haystack, needle)
    sorted_idx = sorted_idx.clamp(max=idx.shape[0] - 1)
    result = idx[sorted_idx]
    result = torch.where(haystack[result] == needle, result, torch.tensor(-1, dtype=result.dtype))
    return result
  # numpy fallback
  idx = np.argsort(haystack)
  sorted_haystack = haystack[idx]
  sorted_idx = np.searchsorted(sorted_haystack, needle)
  idx = np.take(idx, sorted_idx, mode='clip')
  idx[haystack[idx] != needle] = -1
  return idx


def _nvmap(f: Callable[..., Y], *args) -> Y:
  """A vmap that accepts numpy arrays.

  Numpy arrays are statically vmapped, and the elements are passed to f as
  static arguments.  The implication is that all the elements of numpy array
  arguments must be the same.

  Args:
    f: function to be mapped over
    *args: args to be mapped along, passed to f

  Returns:
    the result of vmapping f over args

  Raises:
    RuntimeError: if numpy arg elements do not match
  """
  for arg in args:
    if isinstance(arg, np.ndarray) and arg.shape[0] > 0 and not np.all(arg == arg[0]):
      raise RuntimeError(f'numpy arg elements do not match: {arg}')

  # split out numpy and torch args: numpy arrays become static args,
  # everything else (torch tensors, pytrees, None) goes through vmap.
  # Use shape[0] > 0 (not size > 0) because 2D arrays with shape (n, 0)
  # still need their first row extracted as a representative empty element.
  np_args = []
  for a in args:
    if isinstance(a, np.ndarray):
      np_args.append(a[0] if a.shape[0] > 0 else a)
    else:
      np_args.append(None)
  args = [None if isinstance(a, np.ndarray) else a for a in args]

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
    """Convert numpy scalars/arrays to tensors for vmap compatibility."""
    if isinstance(x, (np.ndarray, np.integer, np.floating)):
      return torch.as_tensor(x)
    return x

  def outer_f(*vmap_inputs, _np_args=np_args, _none_mask=none_mask):
    it = iter(vmap_inputs)
    full_args = [None if m else next(it) for m in _none_mask]
    full_args = [a if n is None else n for n, a in zip(_np_args, full_args)]
    result = f(*full_args)
    return tree_map(_ensure_tensor, result)

  if not vmap_args:
    # No tensor args to vmap over; just call f directly with static args.
    return f(*[n if n is not None else None for n in np_args])

  try:
    return torch.vmap(outer_f, in_dims=tuple([0] * len(vmap_args)))(*vmap_args)
  except ValueError as e:
    if 'NoneType' in str(e):
      # Function returned None for this group (e.g., no free joints).
      return None
    raise


def _check_input(m: Model, args: Any, in_types: str) -> None:
  """Checks that scan input has the right shape."""
  size = {
      'b': m.nbody,
      'j': m.njnt,
      'q': m.nq,
      'v': m.nv,
      'u': m.nu,
      'a': m.na,
      's': m.nsite,
      'c': m.ncam,
  }
  for idx, (arg, typ) in enumerate(zip(args, in_types)):
    if len(arg) != size[typ]:
      raise IndexError(
          (
              f'f argument "{idx}" with type "{typ}" has length "{len(arg)}"'
              f' which does not match the in_types[{idx}] expected length of '
              f'"{size[typ]}".'
          )
      )


def _check_output(
    y: torch.Tensor, take_ids, typ: str, idx: int
) -> None:
  """Checks that scan output has the right shape."""
  if torch.compiler.is_compiling():
    return
  n = take_ids.shape[0] if isinstance(take_ids, (np.ndarray, torch.Tensor)) else len(take_ids)
  if y.shape[0] != n:
    raise IndexError(
        (
            f'f output "{idx}" with type "{typ}" has shape "{y.shape[0]}" '
            f'which does not match the out_types[{idx}] expected size of'
            f' "{n}".'
        )
    )


def _build_flat_cache(m, in_types, out_types, group_by):
  """Precompute grouping indices for scan.flat (numpy-heavy, runs once)."""

  if group_by not in {'j', 'u', 'c'}:
    raise NotImplementedError(f'group by type "{group_by}" not implemented.')

  def key_j(type_ids):
    if any(t in 'jqv' for t in in_types + out_types):
      return tuple(m.jnt_type[type_ids['j']])
    return ()

  def type_ids_j(m, i):
    return {
        'b': i,
        'j': np.nonzero(m.jnt_bodyid == i)[0],
        'v': np.nonzero(m.dof_bodyid == i)[0],
        'q': np.nonzero(_q_bodyid(m) == i)[0],
    }

  def key_u(type_ids):
    ids_u, ids_j = type_ids['u'], type_ids['j']
    return (
        m.actuator_biastype[ids_u],
        m.actuator_gaintype[ids_u],
        m.actuator_dyntype[ids_u],
        m.actuator_trntype[ids_u],
        m.jnt_type[ids_j],
        m.actuator_trnid[ids_u, 1] == -1,
    )

  def type_ids_u(m, i):
    typ_ids = {
        'u': i,
        'a': m.actuator_actadr[i],
        'j': (
            m.actuator_trnid[i, 0]
            if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT)
            else -1
        ),
        's': (
            m.actuator_trnid[i]
            if m.actuator_trntype[i] == TrnType.SITE
            else np.array([-1, -1])
        ),
    }
    v, q = np.array([-1]), np.array([-1])
    if m.actuator_trntype[i] in (TrnType.JOINT, TrnType.JOINTINPARENT):
      v = np.nonzero(m.dof_jntid == typ_ids['j'])[0]
      q = np.nonzero(_q_jointid(m) == typ_ids['j'])[0]
    typ_ids.update({'v': v, 'q': q})
    return typ_ids

  def key_c(type_ids):
    return m.cam_mode[type_ids['c']], m.cam_targetbodyid[type_ids['c']] >= 0

  def type_ids_c(unused_m, i):
    return {'c': i}

  type_ids_fn = {'j': type_ids_j, 'u': type_ids_u, 'c': type_ids_c}[group_by]
  key_fn = {'j': key_j, 'u': key_u, 'c': key_c}[group_by]

  all_types = set(in_types + out_types)
  n_items = {'j': m.nbody, 'u': m.nu, 'c': m.ncam}[group_by]
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
  group_has_output = [
      any(typ_ids[v].size > 0 for v in out_types)
      for _, typ_ids in key_typ_ids
  ]

  flat_ = {'j': 'b', 'u': 'uaj', 'c': 'c'}[group_by]

  # ---- Precompute torch indices for runtime use ----

  # Convert typ_ids to torch.LongTensor for _take operations
  key_typ_ids_torch = []
  for key, typ_ids in key_typ_ids:
    typ_ids_t = {t: _np_to_long(v) for t, v in typ_ids.items()}
    key_typ_ids_torch.append((key, typ_ids_t))

  return {
      'key_typ_ids': key_typ_ids,
      'key_typ_ids_torch': key_typ_ids_torch,
      'group_has_output': group_has_output,
      'order': order,
      'all_types': all_types,
      'flat_': flat_,
  }


def _get_flat_cache(m, in_types, out_types, group_by):
  """Get or build flat scan cache for the given model and type signature."""
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
    group_by: str = 'j',
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
  key_typ_ids_torch = cache['key_typ_ids_torch']
  group_has_output = cache['group_has_output']
  order = cache['order']
  all_types = cache['all_types']
  flat_ = cache['flat_']

  # use cached grouping to take the right data subsets and call vmap(f)
  ys = []
  for (_, typ_ids_t), has_output in zip(key_typ_ids_torch, group_has_output):
    if has_output:
      f_args = [_take(arg, typ_ids_t[typ]) for arg, typ in zip(args, in_types)]
      y = _nvmap(f, *f_args)
      ys.append(y)
    else:
      ys.append(None)

  # remove None results from the final output
  key_typ_ids_torch = [v for y, v in zip(ys, key_typ_ids_torch) if y is not None]
  ys = [y for y in ys if y is not None]
  ys_keys = set([k for k, *_ in key_typ_ids_torch])
  active_order = [o for k, o in order if k in ys_keys]

  # get the original input order
  active_order_per_type = [[o[t] for o in active_order] for t in all_types]
  active_order_per_type = [
      np.concatenate(o) if isinstance(o[0], np.ndarray) else np.array(o)
      for o in active_order_per_type
  ]
  order_dict = dict(zip(all_types, active_order_per_type))

  # concatenate back to a single tree and drop the grouping dimension
  f_ret_is_seq = isinstance(ys[0], (list, tuple))
  ys = ys if f_ret_is_seq else [[y] for y in ys]
  ys = [
      [v if typ in flat_ else torch.flatten(v, 0, 1) for v, typ in zip(y, out_types)]
      for y in ys
  ]
  ys = tree_map(lambda *x: concatenate(x), *ys)

  # put concatenated results back in order
  reordered_ys = []
  for i, (y, typ) in enumerate(zip(ys, out_types)):
    _check_output(y, order_dict[typ], typ, i)
    ids = np.concatenate([np.hstack(v[typ].numpy() if isinstance(v[typ], torch.Tensor) else v[typ]) for _, v in key_typ_ids_torch])
    input_order = order_dict[typ][np.where(order_dict[typ] != -1)]
    reorder_idx = _np_to_long(_index(ids, input_order))
    reordered_ys.append(_take(y, reorder_idx))
  y = reordered_ys if f_ret_is_seq else reordered_ys[0]

  return y


def _build_body_tree_cache(m, in_types, out_types, reverse):
  """Precompute grouping indices for scan.body_tree (numpy-heavy, runs once)."""
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
      if t == 'b':
        continue
      elif t == 'j':
        key += tuple(m.jnt_type[np.nonzero(m.jnt_bodyid == id_)[0]])
      elif t == 'v':
        key += (len(np.nonzero(m.dof_bodyid == id_)[0]),)
      elif t == 'q':
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
      if typ == 'b':
        ids = body_ids
      elif typ == 'j':
        ids = np.stack([np.nonzero(m.jnt_bodyid == b)[0] for b in body_ids])
      elif typ == 'v':
        ids = np.stack([np.nonzero(m.dof_bodyid == b)[0] for b in body_ids])
      elif typ == 'q':
        ids = np.stack([np.nonzero(_q_bodyid(m) == b)[0] for b in body_ids])
      else:
        raise ValueError(f'Unknown in_type: {typ}')
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
        child_info.append((child_key, _np_to_long(id_map), body_ids.size))
      carry_maps[key] = child_info
    elif key in key_parents:
      body_ids_all = [key_body_ids[p] for p in key_parents[key]]
      concat_body_ids = np.concatenate(body_ids_all)
      parent_ids = m.body_parentid[key_body_ids[key]]
      take_idx = _index(concat_body_ids, parent_ids)
      carry_maps[key] = (key_parents[key], _np_to_long(take_idx))

  # Convert in_take indices to torch
  key_in_take_torch = {}
  for key, take_list in key_in_take.items():
    key_in_take_torch[key] = [_np_to_long(ids) for ids in take_list]

  return {
      'key_body_ids': key_body_ids,
      'key_parents': key_parents,
      'key_in_take': key_in_take,
      'key_in_take_torch': key_in_take_torch,
      'key_y_take': key_y_take,
      'keys': keys,
      'carry_maps': carry_maps,
  }


def _get_body_tree_cache(m, in_types, out_types, reverse):
  """Get or build body_tree scan cache."""
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
  key_body_ids = cache['key_body_ids']
  key_in_take_torch = cache['key_in_take_torch']
  key_y_take = cache['key_y_take']
  keys = cache['keys']
  carry_maps = cache['carry_maps']

  # use cached grouping to take the right data subsets and call vmap(f)
  key_y = {}
  for key in keys:
    carry = None

    if reverse:
      for child_key, id_map, n_segments in carry_maps[key]:
        y = key_y[child_key]
        if y is None:
          continue

        def index_sum(x, i=id_map, s=n_segments):
          return segment_sum(x, i, s)

        y = tree_map(index_sum, y)
        carry = y if carry is None else tree_map(torch.add, carry, y)
    elif key in carry_maps:
      parent_keys, take_idx = carry_maps[key]
      ys_all = [key_y[p] for p in parent_keys]
      # Filter out None results from parent groups
      ys_filtered = [y_val for y_val in ys_all if y_val is not None]
      if ys_filtered:
        y = tree_map(lambda *x: concatenate(x), *ys_filtered) if len(ys_filtered) > 1 else ys_filtered[0]
        take_fn = lambda x, i=take_idx: _take(x, i)
        carry = tree_map(take_fn, y)

    f_args = [_take(arg, ids) for arg, ids in zip(args, key_in_take_torch[key])]
    key_y[key] = _nvmap(f, carry, *f_args)

  # slice None results from the final output
  keys = [k for k in keys if key_y[k] is not None]

  # concatenate ys, drop grouping dimensions, put back in order
  y = []
  for i, typ in enumerate(out_types):
    y_typ = [key_y[key] for key in keys]
    if len(out_types) > 1:
      y_typ = [y_[i] for y_ in y_typ]
    if typ != 'b':
      y_typ = tree_map(lambda x: torch.flatten(x, 0, 1), y_typ)
    y_typ = tree_map(lambda *x: torch.cat(x), *y_typ)
    y_take = _np_to_long(np.argsort(np.concatenate([key_y_take[key][i] for key in keys])))
    _check_output(y_typ, y_take, typ, i)
    y.append(_take(y_typ, y_take))

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
  if not isinstance(segment_ids, torch.Tensor):
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
