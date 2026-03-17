"""PRs #175525 + #175852: vmap non-tensor leaves + extension points.

Patches ``torch._functorch.vmap`` to:
- Allow non-tensor return values from vmapped functions (None, int, float,
  bool, complex are broadcast to ``torch.full((batch_size,), value)``).
- Recognise custom "vmappable" container types that implement
  ``_add_batch_dim`` / ``_maybe_remove_batch_dim`` and treat them as leaves
  in pytree operations.

https://github.com/pytorch/pytorch/pull/175525
https://github.com/pytorch/pytorch/pull/175852
"""

from __future__ import annotations

from typing import Any, Callable, NoReturn

import torch
from torch import Tensor
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    _get_name,
    _remove_batch_dim,
    _validate_and_get_batch_size,
    in_dims_t,
    is_batchedtensor,
    out_dims_t,
    tree_flatten,
    tree_unflatten,
    TreeSpec,
    vmap_increment_nesting,
)

# ---------------------------------------------------------------------------
# vmappable registry
# ---------------------------------------------------------------------------
_vmappable_cls_cache: dict[type, bool] = {}


def register_vmappable_cls(cls: type) -> None:
    """Register *cls* as vmap-compatible.

    The class must implement:
      - ``_add_batch_dim(self, in_dim, vmap_level) -> Self``
      - ``_maybe_remove_batch_dim(self, func_name, vmap_level, batch_size, out_dim) -> Self``
      - ``dim(self) -> int``
      - ``size(self, dim) -> int``
    """
    _vmappable_cls_cache[cls] = True


def _is_vmappable(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return False
    cls = type(obj)
    if cls in _vmappable_cls_cache:
        return True
    if hasattr(cls, "_add_batch_dim"):
        _vmappable_cls_cache[cls] = True
        return True
    return False


# ---------------------------------------------------------------------------
# Patched helpers
# ---------------------------------------------------------------------------


def _patched_process_batched_inputs(
    in_dims: in_dims_t, args: tuple[Any, ...], func: Callable[..., Any]
) -> tuple[int, list[int | None], list[Any], TreeSpec]:
    if not isinstance(in_dims, int) and not isinstance(in_dims, tuple):
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...): "
            f"expected `in_dims` to be int or a (potentially nested) tuple "
            f"matching the structure of inputs, got: {type(in_dims)}."
        )
    if len(args) == 0:
        raise ValueError(
            f"vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot "
            f"to add inputs, or you are trying to vmap over a function with no "
            f"inputs. The latter is unsupported."
        )

    flat_args, args_spec = tree_flatten(args, is_leaf=_is_vmappable)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(
            f"vmap({_get_name(func)}, in_dims={in_dims}, ...): "
            f"in_dims is not compatible with the structure of `inputs`. "
            f"in_dims has structure {tree_flatten(in_dims)[1]} but inputs "
            f"has structure {args_spec}."
        )

    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...): "
                f"Got in_dim={in_dim} for an input but in_dim must be either "
                f"an integer dimension or None."
            )
        if (
            isinstance(in_dim, int)
            and not isinstance(arg, Tensor)
            and not _is_vmappable(arg)
        ):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...): "
                f"Got in_dim={in_dim} for an input but the input is of type "
                f"{type(arg)}. We cannot vmap over non-Tensor arguments, "
                f"please use None as the respective in_dim"
            )
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(
                f"vmap({_get_name(func)}, in_dims={in_dims}, ...): "
                f"Got in_dim={in_dim} for some input, but that input is a "
                f"Tensor of dimensionality {arg.dim()} so expected in_dim to "
                f"satisfy -{arg.dim()} <= in_dim < {arg.dim()}."
            )
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()

    return (
        _validate_and_get_batch_size(flat_in_dims, flat_args),
        flat_in_dims,
        flat_args,
        args_spec,
    )


def _patched_create_batched_inputs(
    flat_in_dims: list[int | None],
    flat_args: list[Any],
    vmap_level: int,
    args_spec: TreeSpec,
) -> tuple[Any, ...]:
    batched_inputs = [
        arg
        if in_dim is None
        else (
            arg._add_batch_dim(in_dim=in_dim, vmap_level=vmap_level)
            if _is_vmappable(arg)
            else _add_batch_dim(arg, in_dim, vmap_level)
        )
        for in_dim, arg in zip(flat_in_dims, flat_args)
    ]
    return tree_unflatten(batched_inputs, args_spec)


def _patched_maybe_remove_batch_dim(
    name: str,
    batched_output: Any,
    vmap_level: int,
    batch_size: int,
    out_dim: int | None,
) -> Any:
    if _is_vmappable(batched_output):
        return batched_output._maybe_remove_batch_dim(
            name,
            vmap_level=vmap_level,
            batch_size=batch_size,
            out_dim=out_dim,
        )

    if out_dim is None:
        if isinstance(batched_output, torch.Tensor) and is_batchedtensor(
            batched_output
        ):
            raise ValueError(
                f"vmap({name}, ...): `{name}` can not return a "
                f"BatchedTensor when out_dim is None"
            )
        return batched_output

    if isinstance(batched_output, torch.Tensor):
        return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)

    if batched_output is None:
        return None

    if isinstance(batched_output, (int, float, bool, complex)):
        return torch.full((batch_size,), batched_output)

    raise ValueError(
        f"vmap({name}, ...): `{name}` must only return "
        f"Tensors, got type {type(batched_output)}. "
        "Did you mean to set out_dims= to None for output?"
    )


def _patched_unwrap_batched(
    batched_outputs: Tensor | tuple[Tensor, ...],
    out_dims: out_dims_t,
    vmap_level: int,
    batch_size: int,
    func: Callable[..., Any],
) -> tuple[Any, ...]:
    flat_batched_outputs, output_spec = tree_flatten(
        batched_outputs, is_leaf=_is_vmappable
    )

    def incompatible_error() -> NoReturn:
        raise ValueError(
            f"vmap({_get_name(func)}, ..., out_dims={out_dims}): "
            f"out_dims is not compatible with the structure of `outputs`. "
            f"out_dims has structure {tree_flatten(out_dims)[1]} but outputs "
            f"has structure {output_spec}."
        )

    flat_out_dims: list[int | None] = []
    if isinstance(batched_outputs, torch.Tensor) or _is_vmappable(batched_outputs):
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = list(out_dims)
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        broadcast_result = _broadcast_to_and_flatten(out_dims, output_spec)
        if broadcast_result is None:
            incompatible_error()
        else:
            flat_out_dims = broadcast_result

    flat_outputs = [
        _patched_maybe_remove_batch_dim(
            _get_name(func), batched_output, vmap_level, batch_size, out_dim
        )
        for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)
    ]
    return tree_unflatten(flat_outputs, output_spec)


def _patched_flat_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    batch_size: int,
    flat_in_dims: list[int | None],
    flat_args: list[Any],
    args_spec: TreeSpec,
    out_dims: out_dims_t,
    randomness: str,
    **kwargs: Any,
) -> Any:
    with vmap_increment_nesting(batch_size, randomness) as vmap_level:
        batched_inputs = _patched_create_batched_inputs(
            flat_in_dims, flat_args, vmap_level, args_spec
        )
        batched_outputs = func(*batched_inputs, **kwargs)
        return _patched_unwrap_batched(
            batched_outputs, out_dims, vmap_level, batch_size, func
        )


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------


def apply() -> bool:
    import torch._functorch.vmap as _vmap

    _vmap._vmappable_cls_cache = _vmappable_cls_cache
    _vmap.register_vmappable_cls = register_vmappable_cls
    _vmap._is_vmappable = _is_vmappable
    _vmap._process_batched_inputs = _patched_process_batched_inputs
    _vmap._create_batched_inputs = _patched_create_batched_inputs
    _vmap._maybe_remove_batch_dim = _patched_maybe_remove_batch_dim
    _vmap._unwrap_batched = _patched_unwrap_batched
    _vmap._flat_vmap = _patched_flat_vmap
    return True
