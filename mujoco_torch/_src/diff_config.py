# Copyright 2025 Vincent Moens
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
"""Global configuration for differentiable simulation features.

All differentiability features (smooth collisions, contacts from distance,
adaptive integration) are experimental and disabled by default.  Enable them
via the :func:`differentiable_mode` context manager::

    import mujoco_torch

    with mujoco_torch.differentiable_mode():
        d = mujoco_torch.step(mx, d)

    # Or granularly:
    with mujoco_torch.differentiable_mode(smooth_collisions=True, cfd=False):
        d = mujoco_torch.step(mx, d)

The techniques implemented here are based on:

    Paulus, A., Geist, A. R., Schumacher, P., Musil, V., & Martius, G.
    (2025). Hard Contacts with Soft Gradients: Refining Differentiable
    Simulators for Learning and Control. arXiv:2506.14186.
    https://arxiv.org/abs/2506.14186
"""

import dataclasses
import threading
from contextlib import contextmanager


@dataclasses.dataclass(frozen=True)
class DiffConfig:
    """Configuration for differentiable simulation features.

    Attributes:
      smooth_collisions: Replace discrete case distinctions in collision
          detection with smooth sigmoid-based interpolation.
      cfd: Enable Contacts From Distance — generate informative contact
          gradients for non-colliding bodies using the straight-through trick.
      adaptive_integration: Use adaptive timestep ODE integration to improve
          gradient accuracy during stiff contact phases.
      smooth_sharpness: Sharpness parameter for sigmoid blending in smooth
          collision detection.  Higher = closer to the hard switch.
      cfd_width: Distance threshold for CFD — artificial contact forces are
          generated for signed distances ``0 < r < cfd_width``.
      cfd_dc: Minimum impedance value at the outer edge of the CFD range.
    """

    smooth_collisions: bool = False
    cfd: bool = False
    adaptive_integration: bool = False
    smooth_sharpness: float = 10.0
    cfd_width: float = 0.01
    cfd_dc: float = 0.0
    adaptive_substeps: int = 8
    adaptive_atol: float = 1e-6
    adaptive_rtol: float = 1e-3


_DEFAULT = DiffConfig()
_thread_local = threading.local()


def get_diff_config() -> DiffConfig:
    """Return the active :class:`DiffConfig` for the current thread.

    When no ``differentiable_mode`` context is active the default config
    (all features disabled) is returned.  Under ``torch.compile`` this
    function is evaluated at trace time, so the compiler sees the result
    as a constant and dead-code-eliminates the unused branch.
    """
    return getattr(_thread_local, "diff_config", _DEFAULT)


@contextmanager
def differentiable_mode(
    *,
    smooth_collisions: bool | None = None,
    cfd: bool | None = None,
    adaptive_integration: bool | None = None,
    **kwargs,
):
    """Context manager that activates differentiable simulation features.

    When called with **no arguments**, all three features are enabled.
    Individual features can be toggled explicitly::

        with differentiable_mode(smooth_collisions=True, cfd=False):
            ...

    Extra ``kwargs`` are forwarded to :class:`DiffConfig` (e.g.
    ``smooth_sharpness``, ``cfd_width``).
    """
    all_none = smooth_collisions is None and cfd is None and adaptive_integration is None
    enable_all = all_none and not kwargs

    old = getattr(_thread_local, "diff_config", _DEFAULT)

    overrides: dict = {}
    if enable_all:
        overrides = {
            "smooth_collisions": True,
            "cfd": True,
            "adaptive_integration": True,
        }
    else:
        if smooth_collisions is not None:
            overrides["smooth_collisions"] = smooth_collisions
        if cfd is not None:
            overrides["cfd"] = cfd
        if adaptive_integration is not None:
            overrides["adaptive_integration"] = adaptive_integration
    overrides.update(kwargs)

    new_fields = {f.name: overrides.get(f.name, getattr(old, f.name)) for f in dataclasses.fields(DiffConfig)}
    new = DiffConfig(**new_fields)

    _thread_local.diff_config = new
    try:
        yield new
    finally:
        _thread_local.diff_config = old
