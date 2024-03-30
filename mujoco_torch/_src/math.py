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
"""Some useful math functions."""

import mujoco
import torch


class _CachedConst:
    """A constant tensor created from a Python literal, with per-(dtype,device) caching.

    Avoids CPU→CUDA copies during CUDA graph capture by caching the
    device copy from warmup runs.  Call ``warm_all`` after ``device_put``
    to pre-populate every instance for the target device so that the first
    ``torch.compile`` trace never records a DeviceCopy.
    """

    _instances: list["_CachedConst"] = []

    __slots__ = ("_values", "_cache")

    def __init__(self, values, dtype=None):
        self._values = torch.tensor(values, dtype=dtype)
        self._cache: dict[tuple, torch.Tensor] = {}
        _CachedConst._instances.append(self)

    def get(self, dtype, device) -> torch.Tensor:
        key = (dtype, str(device))
        t = self._cache.get(key)
        if t is None:
            t = self._values.to(dtype=dtype, device=device)
            self._cache[key] = t
        return t

    @classmethod
    def warm_all(cls, device, dtypes=(torch.float64, torch.float32, torch.long, torch.int32, torch.bool)):
        """Pre-populate the cache of every ``_CachedConst`` instance."""
        for inst in cls._instances:
            for dtype in dtypes:
                try:
                    inst.get(dtype, device)
                except Exception:
                    pass


_QUAT_INV_SIGNS = _CachedConst([1, -1, -1, -1])
_TRI_ID = _CachedConst([[0, 3, 4], [3, 1, 5], [4, 5, 2]], dtype=torch.long)
_INF_RANGE = _CachedConst([-torch.inf, torch.inf])


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product of 3-element vectors.

    Drop-in replacement for ``cross`` that avoids the
    ``sizes()`` crash under nested vmap + compile.
    """
    return torch.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        dim=-1,
    )


def safe_div(num: float | torch.Tensor, den: float | torch.Tensor) -> float | torch.Tensor:
    """Safe division for case where denominator is zero."""
    return num / (den + mujoco.mjMINVAL * (den == 0))


def matmul_unroll(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculates a @ b via explicit cell value operations.

    This is faster than generic matmul for small matrices (e.g. 3x3, 4x4).

    Args:
      a: left hand of matmul operand
      b: right hand of matmul operand

    Returns:
      the matrix product of the inputs.
    """
    c = []
    for i in range(a.shape[0]):
        row = []
        for j in range(b.shape[1]):
            s = 0.0
            for k in range(a.shape[1]):
                s += a[i, k] * b[k, j]
            row.append(s)
        c.append(row)

    return torch.stack([torch.stack(row) for row in c])


def norm(x: torch.Tensor, axis: tuple[int, ...] | int | None = None) -> torch.Tensor:
    """Calculates a linalg.norm(x) that's safe for gradients at x=0.

    Avoids a poorly defined gradient for linalg.norm(0) see
    https://github.com/jax-ml/jax/issues/3058 for details
    Args:
      x: A torch.Tensor
      axis: The axis along which to compute the norm

    Returns:
      Norm of the array x.
    """
    is_zero = torch.all(x == 0)
    # temporarily swap x with ones if is_zero, then swap back
    x = torch.where(is_zero, torch.ones_like(x), x)
    n = torch.linalg.norm(x, dim=axis)
    n = torch.where(is_zero, torch.zeros_like(n), n)
    return n


def normalize_with_norm(
    x: torch.Tensor, axis: tuple[int, ...] | int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalizes an array.

    Args:
      x: A torch.Tensor
      axis: The axis along which to compute the norm

    Returns:
      A tuple of (normalized array x, the norm).
    """
    n = norm(x, axis=axis)
    x = x / (n + 1e-6 * (n == 0.0))
    return x, n


def normalize(x: torch.Tensor, axis: tuple[int, ...] | int | None = None) -> torch.Tensor:
    """Normalizes an array.

    Args:
      x: A torch.Tensor
      axis: The axis along which to compute the norm

    Returns:
      normalized array x
    """
    return normalize_with_norm(x, axis=axis)[0]


def rotate(vec: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError("vec must have no batch dimensions.")
    s, u = quat[0], quat[1:]
    r = 2 * (torch.dot(u, vec) * u) + (s * s - torch.dot(u, u)) * vec
    r = r + 2 * s * cross(u, vec)
    return r


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculates the inverse of quaternion q.

    Args:
      q: (4,) quaternion [w, x, y, z]

    Returns:
      The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
    """
    return q * _QUAT_INV_SIGNS.get(q.dtype, q.device)


def quat_sub(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Subtracts two quaternions (u - v) as a 3D velocity."""
    q = quat_mul(quat_inv(v), u)
    axis, angle = quat_to_axis_angle(q)
    return axis * angle


def quat_mul(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return torch.stack(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def quat_mul_axis(q: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Multiplies a quaternion and an axis.

    Args:
      q: (4,) quaternion (w,x,y,z)
      axis: (3,) axis (x,y,z)

    Returns:
      A quaternion q * axis
    """
    return torch.stack(
        [
            -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
            q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
            q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
            q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
        ]
    )


# TODO(erikfrey): benchmark this against brax's quat_to_3x3
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """Converts a quaternion into a 9-dimensional rotation matrix."""
    q = torch.outer(q, q)

    return torch.stack(
        [
            torch.stack(
                [
                    q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3],
                    2 * (q[1, 2] - q[0, 3]),
                    2 * (q[1, 3] + q[0, 2]),
                ]
            ),
            torch.stack(
                [
                    2 * (q[1, 2] + q[0, 3]),
                    q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3],
                    2 * (q[2, 3] - q[0, 1]),
                ]
            ),
            torch.stack(
                [
                    2 * (q[1, 3] - q[0, 2]),
                    2 * (q[2, 3] + q[0, 1]),
                    q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3],
                ]
            ),
        ]
    )


def quat_to_axis_angle(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a quaternion into axis and angle."""
    axis, sin_a_2 = normalize_with_norm(q[1:])
    angle = 2 * torch.arctan2(sin_a_2, q[0])
    angle = torch.where(angle > torch.pi, angle - 2 * torch.pi, angle)

    return axis, angle


def axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Provides a quaternion that describes rotating around axis by angle.

    Args:
      axis: (3,) axis (x,y,z)
      angle: () float angle to rotate by

    Returns:
      A quaternion that rotates around axis by angle
    """
    s, c = torch.sin(angle * 0.5), torch.cos(angle * 0.5)
    return torch.cat([c.unsqueeze(0), axis * s])


def quat_integrate(q: torch.Tensor, v: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Integrates a quaternion given angular velocity and dt."""
    v, norm_ = normalize_with_norm(v)
    angle = dt * norm_
    q_res = axis_angle_to_quat(v, angle)
    q_res = quat_mul(q, q_res)
    return normalize(q_res)


def inert_mul(i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiply inertia by motion, producing force.

    Args:
      i: (10,) inertia (inertia matrix, position, mass)
      v: (6,) spatial motion

    Returns:
      resultant force
    """
    tri_id = _TRI_ID.get(torch.long, i.device)
    inr, pos, mass = i[tri_id], i[6:9], i[9]
    ang = torch.mv(inr, v[:3]) + cross(pos, v[3:])
    vel = mass * v[3:] - cross(pos, v[:3])
    return torch.cat((ang, vel))


def sign(x: torch.Tensor) -> torch.Tensor:
    """Returns the sign of x in the set {-1, 1}."""
    return torch.where(x < 0, -1, 1)


def transform_motion(vel: torch.Tensor, offset: torch.Tensor, rotmat: torch.Tensor):
    """Transform spatial motion.

    Args:
      vel: (6,) spatial motion (3 angular, 3 linear)
      offset: (3,) translation
      rotmat: (3, 3) rotation

    Returns:
      6d spatial velocity
    """
    # TODO(robotics-simulation): are quaternions faster here
    ang, vel = vel[:3], vel[3:]
    vel = rotmat.T @ (vel - cross(offset, ang))
    ang = rotmat.T @ ang
    return torch.cat([ang, vel])


def motion_cross(u, v):
    """Cross product of two motions.

    Args:
      u: (6,) spatial motion
      v: (6,) spatial motion

    Returns:
      resultant spatial motion
    """
    ang = cross(u[:3], v[:3])
    vel = cross(u[3:], v[:3]) + cross(u[:3], v[3:])
    return torch.cat((ang, vel))


def motion_cross_force(v, f):
    """Cross product of a motion and force.

    Args:
      v: (6,) spatial motion
      f: (6,) force

    Returns:
      resultant force
    """
    ang = cross(v[:3], f[:3]) + cross(v[3:], f[3:])
    vel = cross(v[:3], f[3:])
    return torch.cat((ang, vel))


def orthogonals(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns orthogonal vectors `b` and `c`, given a vector `a`."""
    y = torch.eye(3, dtype=a.dtype, device=a.device)[1]
    z = torch.eye(3, dtype=a.dtype, device=a.device)[2]
    b = torch.where((-0.5 < a[1]) & (a[1] < 0.5), y, z)
    b = b - a * a.dot(b)
    # normalize b. however if a is a zero vector, zero b as well.
    b = normalize(b) * torch.any(a)
    return b, cross(a, b)


def make_frame(a: torch.Tensor) -> torch.Tensor:
    """Makes a right-handed 3D frame given a direction."""
    a = normalize(a)
    b, c = orthogonals(a)
    return torch.stack([a, b, c])


# Geometry.


def closest_segment_point(a: torch.Tensor, b: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
    """Returns the closest point on the a-b line segment to a point pt."""
    ab = b - a
    t = torch.dot(pt - a, ab) / (torch.dot(ab, ab) + 1e-6)
    return a + torch.clamp(t, 0.0, 1.0) * ab


def closest_segment_point_and_dist(
    a: torch.Tensor, b: torch.Tensor, pt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns closest point on the line segment and the distance squared."""
    closest = closest_segment_point(a, b, pt)
    dist = (pt - closest).dot(pt - closest)
    return closest, dist


def closest_segment_to_segment_points(
    a0: torch.Tensor, a1: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns closest points between two line segments."""
    # Gets the closest segment points by first finding the closest points
    # between two lines. Points are then clipped to be on the line segments
    # and edge cases with clipping are handled.
    dir_a, len_a = normalize_with_norm(a1 - a0)
    dir_b, len_b = normalize_with_norm(b1 - b0)

    # Segment mid-points.
    half_len_a = len_a * 0.5
    half_len_b = len_b * 0.5
    a_mid = a0 + dir_a * half_len_a
    b_mid = b0 + dir_b * half_len_b

    # Translation between two segment mid-points.
    trans = a_mid - b_mid

    # Parametrize points on each line as follows:
    #  point_on_a = a_mid + t_a * dir_a
    #  point_on_b = b_mid + t_b * dir_b
    # and analytically minimize the distance between the two points.
    dira_dot_dirb = dir_a.dot(dir_b)
    dira_dot_trans = dir_a.dot(trans)
    dirb_dot_trans = dir_b.dot(trans)
    denom = 1 - dira_dot_dirb * dira_dot_dirb

    orig_t_a = (-dira_dot_trans + dira_dot_dirb * dirb_dot_trans) / (denom + 1e-6)
    orig_t_b = dirb_dot_trans + orig_t_a * dira_dot_dirb
    t_a = torch.clamp(orig_t_a, -half_len_a, half_len_a)
    t_b = torch.clamp(orig_t_b, -half_len_b, half_len_b)

    best_a = a_mid + dir_a * t_a
    best_b = b_mid + dir_b * t_b

    # Resolve edge cases where both closest points are clipped to the segment
    # endpoints by recalculating the closest segment points for the current
    # clipped points, and then picking the pair of points with smallest
    # distance. An example of this edge case is when lines intersect but line
    # segments don't.
    new_a, d1 = closest_segment_point_and_dist(a0, a1, best_b)
    new_b, d2 = closest_segment_point_and_dist(b0, b1, best_a)
    best_a = torch.where(d1 < d2, new_a, best_a)
    best_b = torch.where(d1 < d2, best_b, new_b)

    return best_a, best_b


def concatenate(data):
    """Equivalent of jnp.concatenate for PyTorch pytrees."""
    filtered = [x for x in data if x is not None]
    if not filtered:
        return None
    device = None
    for t in filtered:
        if t.device.type != "cpu":
            device = t.device
            break
    if device is not None:
        filtered = [t.to(device) if t.device != device else t for t in filtered]
    return torch.cat(filtered, dim=0)
