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
