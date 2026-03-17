"""Monkey-patches for upstream PyTorch PRs that haven't landed yet.

Each submodule corresponds to one (or a pair of related) upstream PR(s).
Call :func:`apply` at import time to install all patches whose fixes are
not yet present in the running PyTorch build.

The patches are intentionally **no-ops** when the corresponding upstream
change has already been merged, so they are safe to call unconditionally.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_PATCHES = [
    (
        "mujoco_torch.patches._pr175526_while_loop_vmap",
        "PR #175526 (while_loop vmap batching rule)",
    ),
    (
        "mujoco_torch.patches._pr175525_175852_vmap",
        "PR #175525 + #175852 (vmap non-tensor leaves / extension points)",
    ),
    (
        "mujoco_torch.patches._pr176977_meta_converter",
        "PR #176977 (MetaConverter storage memo for wrapper subclasses)",
    ),
]


def apply() -> None:
    """Apply all monkey-patches whose upstream fixes are missing."""
    import importlib

    for module_path, label in _PATCHES:
        mod = importlib.import_module(module_path)
        applied = mod.apply()
        if applied:
            log.info("Applied monkey-patch: %s", label)
        else:
            log.debug("Skipped monkey-patch (already present): %s", label)


def fix_tensordict_unbatched() -> None:
    """Ensure tensordict uses the wrapper-subclass UnbatchedTensor.

    tensordict picks which UnbatchedTensor implementation to use at import
    time by inspecting MetaConverter's source on disk.  Our MetaConverter
    patch modifies the function in memory only, so tensordict's guard sees
    the unpatched source and falls back to the old implementation.

    This function must be called AFTER both :func:`apply` and tensordict
    have been imported.  It re-executes ``tensordict._unbatched`` with the
    guard forced to ``True`` and swaps the ``UnbatchedTensor`` class.
    """
    import tensordict
    import tensordict._unbatched as _ub

    if _ub._HAS_WRAPPER_SUBCLASS_FIX:
        return

    src_path = _ub.__file__
    with open(src_path) as f:
        src = f.read()

    marker = "_HAS_WRAPPER_SUBCLASS_FIX = _has_wrapper_subclass_vmap_fix()"
    if marker not in src:
        log.debug("tensordict._unbatched guard pattern not found, skipping")
        return

    patched_src = src.replace(marker, "_HAS_WRAPPER_SUBCLASS_FIX = True", 1)
    ns: dict = {}
    exec(compile(patched_src, src_path, "exec"), ns)  # noqa: S102

    if not ns.get("_HAS_WRAPPER_SUBCLASS_FIX"):
        log.warning(
            "Failed to activate wrapper-subclass UnbatchedTensor"
        )
        return

    new_cls = ns["UnbatchedTensor"]
    _ub.UnbatchedTensor = new_cls
    _ub._HAS_WRAPPER_SUBCLASS_FIX = True
    tensordict.UnbatchedTensor = new_cls
    log.info(
        "Activated wrapper-subclass UnbatchedTensor "
        "(tensordict._unbatched guard overridden)"
    )
