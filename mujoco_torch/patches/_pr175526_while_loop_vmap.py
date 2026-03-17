"""PR #175526: while_loop vmap batching rule.

Registers a Vmap batching rule for ``while_loop_op`` so that
``torch.vmap`` can batch over functions that contain ``while_loop``.

The rule converts to a masked loop: runs until ALL batch elements have
converged, freezing converged elements via ``torch.where``.

https://github.com/pytorch/pytorch/pull/175526
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.utils._pytree as pytree


def apply() -> bool:
    from torch._higher_order_ops.while_loop import while_loop_op

    transform_type = torch._C._functorch.TransformType.Vmap
    for table_attr in ("py_impls", "functorch_table"):
        if transform_type in getattr(while_loop_op, table_attr, {}):
            return False

    @while_loop_op.py_impl(transform_type)
    def while_loop_vmap(
        interpreter,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple,
        additional_inputs: tuple,
    ):
        from torch._functorch.vmap import restore_vmap, unwrap_batched, wrap_batched

        carried_inputs_list = list(carried_inputs)
        additional_inputs_list = list(additional_inputs)

        (
            (unwrapped_carried, carried_bdims),
            (unwrapped_additional, additional_bdims),
        ) = (
            unwrap_batched(carried_inputs_list, interpreter.level()),
            unwrap_batched(additional_inputs_list, interpreter.level()),
        )

        flat_carried, carried_spec = pytree.tree_flatten(unwrapped_carried)
        flat_carried_bdims, _ = pytree.tree_flatten(carried_bdims)
        flat_additional, additional_spec = pytree.tree_flatten(unwrapped_additional)
        flat_additional_bdims, _ = pytree.tree_flatten(additional_bdims)

        batch_size = interpreter.batch_size()
        randomness = interpreter.randomness()

        flat_carried = [
            t.movedim(bdim, 0) if bdim is not None and bdim != 0 else t
            for t, bdim in zip(flat_carried, flat_carried_bdims, strict=True)
        ]
        flat_carried_bdims = [
            0 if bdim is not None else None for bdim in flat_carried_bdims
        ]

        flat_additional = [
            t.movedim(bdim, 0) if bdim is not None and bdim != 0 else t
            for t, bdim in zip(flat_additional, flat_additional_bdims, strict=True)
        ]
        flat_additional_bdims = [
            0 if bdim is not None else None for bdim in flat_additional_bdims
        ]
        additional_in_dims = tuple(flat_additional_bdims)

        flat_carried = [
            t.unsqueeze(0).expand(batch_size, *t.shape).contiguous()
            if bdim is None
            else t
            for t, bdim in zip(flat_carried, flat_carried_bdims, strict=True)
        ]
        flat_carried_bdims = [0] * len(flat_carried)
        carried_in_dims = tuple(flat_carried_bdims)

        all_in_dims = carried_in_dims + additional_in_dims

        with interpreter.lower():

            def wrapped_cond_fn(*flat_args):
                n_carried = len(flat_carried)
                fn, _per_elem_bdims = restore_vmap(
                    lambda *args: (cond_fn(*args[:n_carried], *args[n_carried:]),),
                    all_in_dims,
                    batch_size,
                    randomness,
                )(*flat_args)
                pred = fn[0]
                return pred.any()

            def wrapped_body_fn(*flat_args):
                n_carried = len(flat_carried)
                carried_args = flat_args[:n_carried]

                cond_result, _ = restore_vmap(
                    lambda *args: (cond_fn(*args[:n_carried], *args[n_carried:]),),
                    all_in_dims,
                    batch_size,
                    randomness,
                )(*flat_args)
                active_mask = cond_result[0]

                new_carried, new_bdims = restore_vmap(
                    lambda *args: body_fn(*args[:n_carried], *args[n_carried:]),
                    all_in_dims,
                    batch_size,
                    randomness,
                )(*flat_args)

                flat_new, _new_spec = pytree.tree_flatten(new_carried)
                flat_new_bdims, _ = pytree.tree_flatten(new_bdims)

                flat_new = [
                    t.movedim(bdim, 0) if bdim is not None and bdim != 0 else t
                    for t, bdim in zip(flat_new, flat_new_bdims, strict=True)
                ]
                flat_new = [
                    t.unsqueeze(0).expand(batch_size, *t.shape)
                    if bdim is None
                    else t
                    for t, bdim in zip(flat_new, flat_new_bdims, strict=True)
                ]

                mask = active_mask
                result = []
                for old, new in zip(carried_args, flat_new, strict=True):
                    shape = (batch_size,) + (1,) * (old.ndim - 1)
                    result.append(torch.where(mask.view(shape), new, old))
                return tuple(result)

            unwrapped_out = while_loop_op(
                wrapped_cond_fn,
                wrapped_body_fn,
                tuple(flat_carried),
                tuple(flat_additional),
            )

        out_bdims = (0,) * len(unwrapped_out)
        return wrap_batched(unwrapped_out, out_bdims, interpreter.level())

    return True
