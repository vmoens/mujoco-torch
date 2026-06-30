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
    import torch

    if issubclass(_ub.UnbatchedTensor, torch.Tensor):
        return

    # Older tensordict versions expose this guard; newer versions have the fix
    # unconditionally and no longer carry the flag.  Treat absence as "fix
    # already present" and skip the re-exec dance.
    if getattr(_ub, "_HAS_WRAPPER_SUBCLASS_FIX", True):
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
        log.warning("Failed to activate wrapper-subclass UnbatchedTensor")
        return

    new_cls = ns["UnbatchedTensor"]
    _ub.UnbatchedTensor = new_cls
    _ub._HAS_WRAPPER_SUBCLASS_FIX = True
    tensordict.UnbatchedTensor = new_cls
    log.info("Activated wrapper-subclass UnbatchedTensor (tensordict._unbatched guard overridden)")


def fix_unbatched_tensor_vmap() -> None:
    """Keep ``UnbatchedTensor`` payloads unbatched across stack/vmap.

    TensorDict main includes the upstream fix for this behavior
    (pytorch/tensordict#1730).  Older/stable TensorDict releases can still be
    missing some or all of the same metadata handling, so patch only when the
    installed ``UnbatchedTensor`` does not already preserve stack metadata.
    """
    import warnings

    import tensordict
    import torch

    cls = tensordict.UnbatchedTensor

    if _has_unbatched_stack_metadata(cls, torch):
        return

    if not hasattr(cls, "batch_size"):

        def _get_batch_size(self):
            return getattr(self, "_batch_size", torch.Size())

        def _set_batch_size(self, batch_size):
            self._batch_size = torch.Size(batch_size)

        cls.batch_size = property(_get_batch_size, _set_batch_size)

    def _with_batch_size(self, batch_size):
        if hasattr(self, "_with_batch_size"):
            return self._with_batch_size(batch_size)
        if hasattr(self, "copy"):
            out = self.copy()
        elif hasattr(self, "_data"):
            out = type(self)(self._data)
        else:
            out = self.clone()
        out.batch_size = batch_size
        return out

    def _add_batch_dim(self, *, in_dim: int, vmap_level: int):
        batch_size = list(self.batch_size)
        if in_dim < 0 and batch_size:
            in_dim %= len(batch_size)
        return _with_batch_size(self, batch_size[:in_dim] + batch_size[in_dim + 1 :])

    def _maybe_remove_batch_dim(
        self,
        funcname=None,  # noqa: ANN001
        *,
        vmap_level: int,
        batch_size: int,
        out_dim: int | None,
    ):
        if out_dim is None:
            return self
        current_batch_size = list(self.batch_size)
        if out_dim < 0:
            out_dim %= len(current_batch_size) + 1
        current_batch_size.insert(out_dim, batch_size)
        return _with_batch_size(self, current_batch_size)

    def _stack_non_tensor(cls_, list_of_non_tensor, dim: int = 0, raise_if_non_unique=False):
        first = list_of_non_tensor[0]

        def _ptr(value):
            if hasattr(value, "data_ptr"):
                return value.data_ptr()
            data = getattr(value, "data", None)
            if hasattr(data, "data_ptr"):
                return data.data_ptr()
            return None

        ptr = _ptr(first)
        if ptr is not None and any(_ptr(other) != ptr for other in list_of_non_tensor[1:]):
            warnings.warn(
                "Stacking UnbatchedTensors with different data storage. "
                "Only the first element's data will be kept. "
                "UnbatchedTensor is shape-invariant; if you need different data "
                "per batch element, consider using a regular tensor.",
                stacklevel=2,
            )

        batch_size = list(first.batch_size)
        if dim < 0:
            dim %= len(batch_size) + 1
        batch_size.insert(dim, len(list_of_non_tensor))
        return _with_batch_size(first, batch_size)

    cls._add_batch_dim = _add_batch_dim
    cls._maybe_remove_batch_dim = _maybe_remove_batch_dim
    cls._stack_non_tensor = classmethod(_stack_non_tensor)


def _has_unbatched_stack_metadata(cls, torch) -> bool:
    """Return True when TensorDict already has pytorch/tensordict#1730."""
    if not (issubclass(cls, torch.Tensor) and hasattr(cls, "batch_size") and hasattr(cls, "_with_batch_size")):
        return False
    try:
        value = cls(torch.zeros((), dtype=torch.int32))
        stacked = cls._stack_non_tensor([value, value], dim=0)
    except Exception:
        return False
    return getattr(stacked, "batch_size", None) == torch.Size([2]) and stacked.shape == torch.Size([])
