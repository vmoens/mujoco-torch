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
"""MjTensorClass: TensorClass base for mujoco-torch data structures."""

import copy
import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorClass
from tensordict.tensorclass import _TensorClassMeta


def _tree_replace(
    base: "MjTensorClass",
    attr: Sequence[str],
    val: torch.Tensor | None,
) -> "MjTensorClass":
    """Sets attributes in a struct.dataclass with values."""
    if not attr:
        return base

    # special case for List attribute
    if len(attr) > 1 and isinstance(getattr(base, attr[0]), list):
        lst = copy.deepcopy(getattr(base, attr[0]))

        for i, g in enumerate(lst):
            if not hasattr(g, attr[1]):
                continue
            v = val if not hasattr(val, "__iter__") else val[i]
            lst[i] = _tree_replace(g, attr[1:], v)

        return base.replace(**{attr[0]: lst})

    if len(attr) == 1:
        return base.replace(**{attr[0]: val})

    return base.replace(**{attr[0]: _tree_replace(getattr(base, attr[0]), attr[1:], val)})


def _collect_conflicting_attrs(bases):
    """Collect attribute names from bases that would conflict with dataclass fields.

    Any non-underscore attribute (method, property, descriptor) inherited from
    a base class looks like a "default" to the dataclass machinery and would
    cause field-ordering errors when a later field has no default.
    """
    attrs = set()
    for base in bases:
        for cls in base.__mro__:
            for attr_name, _ in cls.__dict__.items():
                if not attr_name.startswith("_"):
                    attrs.add(attr_name)
    return attrs


class _MjMeta(_TensorClassMeta):
    """Metaclass that shadows conflicting TensorClass attributes before
    the dataclass decorator runs, so that field names like ``size``, ``grad``,
    ``names``, or ``dim`` don't trigger ordering errors."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        if any(getattr(b, "_is_mj_base", False) for b in bases):
            conflicting = _collect_conflicting_attrs(bases)
            annotations = namespace.get("__annotations__", {})
            for field_name in annotations:
                if field_name not in namespace and field_name in conflicting:
                    namespace[field_name] = dataclasses.field()
        return super().__new__(mcs, name, bases, namespace, **kwargs)


class MjTensorClass(
    TensorClass["nocast", "shadow"],
    metaclass=_MjMeta,
):
    """TensorClass base for mujoco-torch data structures.

    Provides backward-compatible ``replace()`` and ``tree_replace()`` methods
    that mirror the PyTreeNode API.

    Uses ``shadow`` mode and a custom metaclass so that field names like
    ``size``, ``grad``, or ``dim`` do not conflict with TensorDict's
    reserved attributes.
    """

    _is_mj_base = True

    def replace(self, **kwargs: Any) -> "MjTensorClass":
        """Return a shallow copy with the given fields replaced."""
        clone = self.clone(recurse=False)
        # Write directly to the internal dict to avoid per-key _set_str calls.
        # The standard setattr path goes through __setattr__ → _set → set_tensor
        # → _set_str, each of which becomes a separate dynamo frame that guards
        # on the key string. Using a single dict.update() avoids O(N)
        # recompilations for N unique keys.
        clone._tensordict._tensordict.update(kwargs)
        return clone

    def update_(self, **kwargs: Any) -> "MjTensorClass":
        """Update fields **in-place** (no shallow copy).

        Faster than :meth:`replace` because it skips the ``clone(recurse=False)``
        allocation (80-field dict copy).  The caller is responsible for ensuring
        that no other live reference to this object relies on the old field values.
        """
        self._tensordict._tensordict.update(kwargs)
        return self

    def tree_replace(
        self,
        params: dict[str, torch.Tensor | None],
    ) -> "MjTensorClass":
        """Replace nested fields using dot-separated key paths."""
        new = self
        for k, v in params.items():
            new = _tree_replace(new, k.split("."), v)
        return new

    @classmethod
    def fields(cls) -> tuple[dataclasses.Field, ...]:
        return dataclasses.fields(cls)


# NOTE: Field-name / TensorClass-property collisions (e.g. Model.names)
# are resolved with a targeted property descriptor installed on the specific
# subclass — see types.py after the Model definition.  We intentionally do
# NOT install a custom __getattribute__ because any override on a class
# causes graph breaks in torch.compile on *every* attribute access, which
# cascades into O(N) recompilations for N unique field names.
