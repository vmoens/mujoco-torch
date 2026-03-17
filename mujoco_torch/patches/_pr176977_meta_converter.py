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
import re


def apply() -> bool:
    import torch._subclasses.meta_utils as _mu

    src = inspect.getsource(_mu.MetaConverter.meta_tensor)

    # The marker identifies the storage block we need to guard.
    marker = "s = t.storage\n"
    if marker not in src:
        return False
    # Check for the specific guard pattern from PR #176977: the storage block
    # must be wrapped with ``if not t.is_traceable_wrapper_subclass:``.
    # A simple substring check is too broad because nightly may already
    # contain is_traceable_wrapper_subclass in *other* code paths.
    if re.search(
        r"not\s+t\.is_traceable_wrapper_subclass.*\n\s+s = t\.storage", src
    ):
        return False

    # Find the ``r.set_()`` call and dynamically compute its indentation so we
    # can wrap it in a ``try``/``except`` regardless of surrounding indent level.
    set_call = "r.set_(r_s, storage_offset, sizes, strides)"
    if set_call not in src:
        return False

    # Determine the exact leading whitespace for the r.set_() line.
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(set_call):
            indent = len(line) - len(stripped)
            break
    else:
        return False

    # Determine the indentation of the ``def meta_tensor`` line to know how
    # much to strip when compiling the function as standalone code.
    first_line = src.splitlines()[0]
    def_indent = len(first_line) - len(first_line.lstrip())

    # Build replacement: wrap ``r.set_()`` in try/except to swallow the
    # cross-device storage error that occurs for wrapper subclasses.
    i0 = " " * indent  # indentation of the original r.set_ line
    i1 = " " * (indent + 4)
    i2 = " " * (indent + 8)
    old = i0 + set_call
    new = (
        f"{i0}try:\n"
        f"{i1}{set_call}\n"
        f"{i0}except RuntimeError as __e:\n"
        f'{i1}if "storage" not in str(__e) or "device" not in str(__e):\n'
        f"{i2}raise"
    )
    patched_src = src.replace(old, new, 1)

    # Strip the class-body indentation so the ``def`` starts at column 0.
    if def_indent > 0:
        stripped_lines = []
        for line in patched_src.splitlines(True):
            if line.strip():
                stripped_lines.append(line[def_indent:])
            else:
                stripped_lines.append(line)
        patched_src = "".join(stripped_lines)

    ns: dict = {}
    code = compile(patched_src, _mu.__file__, "exec")
    exec(code, _mu.__dict__, ns)  # noqa: S102
    _mu.MetaConverter.meta_tensor = ns["meta_tensor"]
    return True
