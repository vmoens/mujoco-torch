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
"""Control Moment Gyro (CMG) geometry and steering math.

Tooling to score CMG cluster configurations by their proximity to
internal singularities. Used by the satellite environments under
:mod:`mujoco_torch.zoo.satellite` and by downstream RL libraries that
penalize singular CMG configurations during training.

Conventions:

* Gimbal axes ``g_i`` are unit vectors fixed in the body frame.
* Rotor axes ``r_i(theta_i)`` rotate around the gimbal axis by the
  gimbal angle ``theta_i``; at ``theta_i = 0`` the rotor axis must be
  orthogonal to the gimbal axis.
* Each CMG with rotor angular momentum ``h`` produces a torque
  ``tau_i = h * (g_i x r_i(theta_i))`` per unit gimbal rate; the
  cluster Jacobian stacks these as columns.

Functions are batched: ``gimbal_angles`` of shape ``(..., N)`` produces
a Jacobian of shape ``(..., 3, N)``.
"""
from __future__ import annotations

import math

import torch


def rodrigues_rotate(
    axis: torch.Tensor, vector: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """Rotate ``vector`` around unit ``axis`` by ``theta`` (Rodrigues).

    Args:
        axis: ``(3, N)`` unit vectors, one rotation axis per column.
        vector: ``(3, N)`` vectors to rotate, one per column.
        theta: ``(..., N)`` rotation angles in radians.

    Returns:
        ``(..., 3, N)`` rotated vectors. The leading ``...`` dims of
        ``theta`` broadcast over the rotation operation.
    """
    cos_t = torch.cos(theta).unsqueeze(-2)
    sin_t = torch.sin(theta).unsqueeze(-2)
    axis_dot_vec = (axis * vector).sum(dim=0)
    axis_cross_vec = torch.linalg.cross(axis, vector, dim=0)
    return cos_t * vector + sin_t * axis_cross_vec + (1.0 - cos_t) * axis_dot_vec * axis


def cmg_jacobian(
    gimbal_angles: torch.Tensor,
    gimbal_axes: torch.Tensor,
    rotor_axes_ref: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Output-torque Jacobian of a CMG cluster over gimbal rates.

    For each CMG, the contribution to body torque per unit gimbal rate
    is ``h * (g_i x r_i(theta_i))``. The columns of the returned matrix
    are the per-CMG contributions; the rows index body-frame torque
    components.

    Args:
        gimbal_angles: ``(..., N)`` current gimbal angles, radians.
        gimbal_axes: ``(3, N)`` fixed body-frame gimbal axes, unit norm.
        rotor_axes_ref: ``(3, N)`` rotor axes at ``theta = 0``, unit
            norm, perpendicular to the corresponding gimbal axis.
        h: scalar rotor angular momentum magnitude.

    Returns:
        ``(..., 3, N)`` torque Jacobian.
    """
    rotor = rodrigues_rotate(gimbal_axes, rotor_axes_ref, gimbal_angles)
    g_expanded = gimbal_axes.expand_as(rotor)
    return h * torch.linalg.cross(g_expanded, rotor, dim=-2)


def manipulability(jac: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Yoshikawa manipulability ``sqrt(det(J J^T) + eps)``.

    Approaches zero as the Jacobian loses rank (cluster becomes
    singular); the ``+eps`` guard keeps the metric finite at exact
    singularities, suitable for use in a reward shaping term.

    Args:
        jac: ``(..., 3, N)`` Jacobian, with ``N >= 3``.
        eps: small floor added inside the square root.

    Returns:
        ``(...,)`` manipulability per batch element.
    """
    jjt = jac @ jac.transpose(-1, -2)
    det = torch.linalg.det(jjt).clamp_min(0.0)
    return torch.sqrt(det + eps)


def pyramid_4cmg_geometry(
    skew_deg: float = 54.7356,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard 4-CMG pyramid: gimbal axes tilted by ``skew_deg`` from +z.

    The default skew angle ``arctan(sqrt(2)) ~ 54.74 deg`` produces a
    spherical momentum envelope (the textbook configuration; see Wie,
    "Space Vehicle Dynamics and Control", 2008).

    Args:
        skew_deg: pyramid skew angle in degrees.
        device: device for the returned tensors.
        dtype: dtype for the returned tensors.

    Returns:
        ``(gimbal_axes, rotor_axes_ref)``, both shaped ``(3, 4)``.
    """
    beta = math.radians(skew_deg)
    cb, sb = math.cos(beta), math.sin(beta)
    gimbal_axes = torch.tensor(
        [
            [sb, 0.0, -sb, 0.0],
            [0.0, sb, 0.0, -sb],
            [cb, cb, cb, cb],
        ],
        device=device,
        dtype=dtype,
    )
    # Reference rotor axes lie in the body xy-plane, orthogonal to the
    # corresponding gimbal axis projection.
    rotor_axes_ref = torch.tensor(
        [
            [0.0, 1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    return gimbal_axes, rotor_axes_ref


def orthogonal_6cmg_geometry(
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """6-CMG redundant cluster with gimbal axes along ``+/-x, +/-y, +/-z``.

    Reference rotor axes lie in the plane perpendicular to each gimbal
    axis, chosen so the cluster is full-rank at ``theta = 0``.

    Args:
        device: device for the returned tensors.
        dtype: dtype for the returned tensors.

    Returns:
        ``(gimbal_axes, rotor_axes_ref)``, both shaped ``(3, 6)``.
    """
    gimbal_axes = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    rotor_axes_ref = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    return gimbal_axes, rotor_axes_ref
