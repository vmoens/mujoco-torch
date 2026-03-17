"""PR #176977: Skip storage memo for wrapper subclasses in MetaConverter.

``_make_wrapper_subclass`` tensor subclasses crash with a cross-device
storage error when used as non-batched inputs to ``torch.vmap`` inside
``torch.compile``.  The root cause is that the wrapper's placeholder storage
(which carries the original device, e.g. ``cuda:0``) gets entered into the
``MetaConverter`` storage memo.  On a later encounter the "crazy town"
fallback calls ``r.set_(meta_storage, …)`` under ``in_kernel_invocation_manager``
(which disables ``__torch_dispatch__``), and the C++ kernel raises because the
tensor device (``cuda:0``) differs from the storage device (``meta``).

The fix guards the entire storage-memo block with
``if not t.is_traceable_wrapper_subclass``.

https://github.com/pytorch/pytorch/pull/176977
"""

from __future__ import annotations

import inspect
import textwrap


def apply() -> bool:
    import torch._subclasses.meta_utils as _mu

    src = inspect.getsource(_mu.MetaConverter.meta_tensor)
    marker = "                    s = t.storage\n"
    if marker not in src:
        return False
    if "is_traceable_wrapper_subclass" in src:
        return False

    src = textwrap.dedent(src)
    # After dedent the marker loses its leading whitespace relative to the
    # class body (4 spaces removed).  Recompute it.
    dedented_marker = marker.lstrip()
    # The storage block lives inside an ``else:`` branch that is indented 20
    # spaces in the original source.  After dedenting by 4 (class body) the
    # block starts at 16 spaces.  We need to wrap lines 16-deep and deeper
    # in a ``if not t.is_traceable_wrapper_subclass:`` guard and indent them
    # by 4.
    #
    # Rather than reindenting dozens of lines (brittle across torch versions),
    # we apply a surgical fix: catch the specific ``RuntimeError`` from the
    # ``r.set_()`` call that triggers the cross-device crash and silently
    # skip the storage update for wrapper subclasses.
    old = "r.set_(r_s, storage_offset, sizes, strides)"
    if old not in src:
        return False

    new = (
        "try:\n"
        + " " * 40
        + "r.set_(r_s, storage_offset, sizes, strides)\n"
        + " " * 36
        + "except RuntimeError as __e:\n"
        + " " * 40
        + 'if "storage" not in str(__e) or "device" not in str(__e):\n'
        + " " * 44
        + "raise"
    )
    patched_src = src.replace(old, new, 1)

    ns: dict = {}
    code = compile(patched_src, _mu.__file__, "exec")
    exec(code, _mu.__dict__, ns)  # noqa: S102
    _mu.MetaConverter.meta_tensor = ns["meta_tensor"]
    return True
