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
"""Pure-PyTorch ray-cast renderer."""

import numpy as np
import torch

from mujoco_torch._src.ray import _PRIMITIVE_RAY_FUNC, ray_precomputed
from mujoco_torch._src.types import Data, GeomType, Model


def precompute_render_data(m: Model) -> tuple:
    """Pre-compute scene geometry data for rendering.

    Call once per model and pass the result to :func:`render` via the
    ``precomp`` keyword to avoid repeated numpy work on every frame.

    Args:
      m: MJX Model.

    Returns:
      A tuple of ``(fn, id_t, geom_size_t, vis_filter_t)`` entries, one per
      primitive geom type present in the scene.  The tensors are plain CPU
      tensors; :func:`render` moves them to the data device before use.
    """
    geom_matid = np.asarray(m.geom_matid.cpu())
    geom_rgba = np.asarray(m.geom_rgba.cpu())
    geom_size = np.asarray(m.geom_size.cpu())
    geom_type = np.asarray(m.geom_type)

    mat_rgba = np.asarray(m.mat_rgba.cpu()) if m.mat_rgba.shape[0] > 0 else np.empty((0, 4))

    visible = (geom_matid != -1) | (geom_rgba[:, 3] != 0)
    if mat_rgba.shape[0] > 0:
        visible = visible & ((geom_matid == -1) | (mat_rgba[np.clip(geom_matid, 0, None), 3] != 0))

    entries = []
    for gt, fn in _PRIMITIVE_RAY_FUNC.items():
        (id_np,) = np.nonzero(geom_type == gt)
        if id_np.size == 0:
            continue
        id_t = torch.tensor(id_np, dtype=torch.long)
        size_t = torch.as_tensor(geom_size[id_np].copy(), dtype=torch.float64)
        vis_t = torch.tensor(visible[id_np])
        entries.append((fn, id_t, size_t, vis_t))

    return tuple(entries)


def _resolve_precomp(precomp: tuple, device: torch.device) -> tuple:
    """Move precomp tensors to the target device."""
    resolved = []
    for fn, id_t, size_t, vis_t in precomp:
        resolved.append(
            (
                fn,
                id_t.to(device),
                size_t.to(device),
                vis_t.to(device),
            )
        )
    return tuple(resolved)


def _generate_rays(
    cam_xpos: torch.Tensor,
    cam_xmat: torch.Tensor,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray origins and directions for every pixel.

    Uses the MuJoCo camera convention: -z forward, +x right, +y up.

    Args:
      cam_xpos: camera world position ``(3,)``.
      cam_xmat: camera rotation matrix, local-to-world ``(3, 3)``.
      fovy_deg: vertical field of view in degrees.
      width: image width in pixels.
      height: image height in pixels.

    Returns:
      ``(origins, directions)`` each of shape ``(H, W, 3)``.
    """
    device = cam_xpos.device
    dtype = cam_xpos.dtype

    aspect = width / height
    half_h = torch.tan(torch.tensor(fovy_deg * (torch.pi / 360.0), dtype=dtype, device=device))
    half_w = half_h * aspect

    u = torch.linspace(0.5, width - 0.5, width, dtype=dtype, device=device)
    v = torch.linspace(0.5, height - 0.5, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")

    dirs_cam = torch.stack(
        [
            (2 * u / width - 1) * half_w,
            (1 - 2 * v / height) * half_h,
            -torch.ones_like(u),
        ],
        dim=-1,
    )
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)

    dirs_world = (cam_xmat @ dirs_cam.unsqueeze(-1)).squeeze(-1)

    origins = cam_xpos.broadcast_to(height, width, 3)
    return origins, dirs_world


def _geom_color(
    m: Model,
    geom_ids: torch.Tensor,
) -> torch.Tensor:
    """Look up RGBA colour for hit geom ids.

    Args:
      m: Model with ``geom_rgba``, ``mat_rgba``, ``geom_matid``.
      geom_ids: ``(...)`` int tensor of geom indices (``-1`` = miss).

    Returns:
      ``(..., 4)`` RGBA tensor.
    """
    safe_ids = geom_ids.clamp(min=0)
    geom_color = m.geom_rgba[safe_ids]

    if m.mat_rgba.shape[0] > 0:
        mat_ids = m.geom_matid[safe_ids]
        has_mat = mat_ids >= 0
        mat_color = m.mat_rgba[mat_ids.clamp(min=0)]
        color = torch.where(has_mat.unsqueeze(-1), mat_color, geom_color)
    else:
        color = geom_color

    miss = (geom_ids < 0).unsqueeze(-1)
    color = torch.where(miss, torch.zeros_like(color), color)
    return color


def _safe_normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize vectors, returning zero for degenerate inputs."""
    return v / v.norm(dim=dim, keepdim=True).clamp(min=1e-10)


def _compute_normals(
    hit_points: torch.Tensor,
    geom_ids: torch.Tensor,
    m: Model,
    d: Data,
) -> torch.Tensor:
    """Compute world-frame surface normals at hit points.

    Args:
      hit_points: ``(N, 3)`` world-frame positions.
      geom_ids: ``(N,)`` int tensor (``-1`` for miss).
      m: Model.
      d: Data.

    Returns:
      ``(N, 3)`` unit normals (zero for misses).
    """
    safe_ids = geom_ids.clamp(min=0)

    geom_xpos = d.geom_xpos[safe_ids]
    geom_xmat = d.geom_xmat[safe_ids]
    geom_size = m.geom_size[safe_ids]
    geom_type_t = torch.as_tensor(
        np.asarray(m.geom_type),
        device=hit_points.device,
    )[safe_ids]

    hit_local = (geom_xmat.transpose(-1, -2) @ (hit_points - geom_xpos).unsqueeze(-1)).squeeze(-1)

    # --- per-type normals in local frame ---

    # Plane: always (0, 0, 1)
    n_plane = torch.zeros_like(hit_local)
    n_plane[..., 2] = 1.0

    # Sphere: normalize(hit_local)
    n_sphere = _safe_normalize(hit_local)

    # Ellipsoid: normalize(hit_local / size^2)
    n_ellipsoid = _safe_normalize(hit_local / geom_size.square().clamp(min=1e-10))

    # Box: face normal from the axis with largest |hit_local / size|
    abs_scaled = hit_local.abs() / geom_size.clamp(min=1e-10)
    face_idx = abs_scaled.argmax(dim=-1, keepdim=True)
    n_box = torch.zeros_like(hit_local)
    n_box.scatter_(
        -1,
        face_idx,
        torch.sign(hit_local).gather(-1, face_idx),
    )

    # Capsule: cap vs cylinder
    cap_z = torch.sign(hit_local[..., 2]) * geom_size[..., 1]
    cap_center = torch.zeros_like(hit_local)
    cap_center[..., 2] = cap_z
    n_cap = _safe_normalize(hit_local - cap_center)
    n_cyl = torch.zeros_like(hit_local)
    n_cyl[..., :2] = _safe_normalize(hit_local[..., :2], dim=-1)
    on_cap = hit_local[..., 2].abs() > geom_size[..., 1]
    n_capsule = torch.where(on_cap.unsqueeze(-1), n_cap, n_cyl)

    # --- select by geom type ---
    n_local = n_plane
    n_local = torch.where(
        (geom_type_t == int(GeomType.SPHERE)).unsqueeze(-1),
        n_sphere,
        n_local,
    )
    n_local = torch.where(
        (geom_type_t == int(GeomType.CAPSULE)).unsqueeze(-1),
        n_capsule,
        n_local,
    )
    n_local = torch.where(
        (geom_type_t == int(GeomType.ELLIPSOID)).unsqueeze(-1),
        n_ellipsoid,
        n_local,
    )
    n_local = torch.where(
        (geom_type_t == int(GeomType.BOX)).unsqueeze(-1),
        n_box,
        n_local,
    )

    # transform to world frame
    n_world = (geom_xmat @ n_local.unsqueeze(-1)).squeeze(-1)

    miss = (geom_ids < 0).unsqueeze(-1)
    return torch.where(miss, torch.zeros_like(n_world), n_world)


def _shade(
    normals: torch.Tensor,
    hit_points: torch.Tensor,
    view_dirs: torch.Tensor,
    base_color: torch.Tensor,
    m: Model,
    d: Data,
) -> torch.Tensor:
    """Apply Lambert diffuse + Phong specular shading.

    Args:
      normals: ``(N, 3)`` world-frame surface normals.
      hit_points: ``(N, 3)`` world-frame hit positions.
      view_dirs: ``(N, 3)`` normalised ray directions (camera → hit).
      base_color: ``(N, 3)`` unlit surface RGB.
      m: Model with light properties.
      d: Data with ``light_xpos`` / ``light_xdir``.

    Returns:
      ``(N, 3)`` shaded RGB clamped to [0, 1].
    """
    shaded = torch.zeros_like(base_color)

    for i in range(m.nlight):
        light_type_i = int(m.light_type[i])

        # direction from hit point towards the light source
        to_light_dir = -d.light_xdir[i].expand_as(hit_points)
        to_light_pos = d.light_xpos[i] - hit_points
        to_light = torch.where(
            torch.tensor(light_type_i == 1, device=hit_points.device).unsqueeze(-1),
            to_light_dir,
            to_light_pos,
        )
        to_light = _safe_normalize(to_light)

        light_diff = m.light_diffuse[i]
        light_amb = m.light_ambient[i]
        light_spec = m.light_specular[i]

        # Lambert diffuse
        ndotl = (normals * to_light).sum(dim=-1, keepdim=True).clamp(min=0)
        diffuse = base_color * light_diff * ndotl

        # Phong specular: R = 2(N·L)N - L
        reflect = 2 * ndotl * normals - to_light
        reflect = _safe_normalize(reflect)
        rdotv = (
            (-view_dirs * reflect)
            .sum(
                dim=-1,
                keepdim=True,
            )
            .clamp(min=0)
        )
        specular = light_spec * rdotv.pow(50)

        ambient = base_color * light_amb

        shaded = shaded + diffuse + specular + ambient

    return shaded.clamp(0, 1)


def render(
    m: Model,
    d: Data,
    camera_id: int = 0,
    width: int = 64,
    height: int = 64,
    precomp: tuple | None = None,
    shading: bool = True,
    background: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch ray-cast render.

    Casts one ray per pixel through the scene using the existing
    :func:`~mujoco_torch._src.ray.ray_precomputed` infrastructure and returns
    shaded (or flat) RGB, depth and segmentation buffers.

    Args:
      m: MJX Model.
      d: MJX Data with up-to-date ``cam_xpos`` / ``cam_xmat`` (call
         :func:`~mujoco_torch.kinematics` or :func:`~mujoco_torch.step`
         first).
      camera_id: index of the camera to render from.
      width: output image width in pixels.
      height: output image height in pixels.
      precomp: pre-computed render data from :func:`precompute_render_data`.
         Computed on-the-fly when *None*.
      shading: when *True* and the model has lights, apply Lambert diffuse +
         Phong specular shading.  Set to *False* for flat-colour output.
      background: RGB tuple in [0, 1] for pixels that miss all geometry.
         Defaults to black ``(0, 0, 0)`` when *None*.

    Returns:
      ``(rgb, depth, seg)`` where

      * **rgb** is ``(H, W, 3)`` float in [0, 1],
      * **depth** is ``(H, W)`` float (``-1`` for no hit),
      * **seg** is ``(H, W)`` long (geom id, ``-1`` for no hit).
    """
    if precomp is None:
        precomp = precompute_render_data(m)

    device = d.geom_xpos.device
    precomp = _resolve_precomp(precomp, device)

    cam_xpos = d.cam_xpos[camera_id]
    cam_xmat = d.cam_xmat[camera_id]
    fovy_deg = float(m.cam_fovy[camera_id])

    origins, directions = _generate_rays(cam_xpos, cam_xmat, fovy_deg, width, height)

    origins_flat = origins.reshape(-1, 3)
    dirs_flat = directions.reshape(-1, 3)

    dists, geom_ids = torch.vmap(
        lambda p, v: ray_precomputed(precomp, d, p, v),
    )(origins_flat, dirs_flat)

    depth = dists.reshape(height, width)
    seg = geom_ids.reshape(height, width)

    rgba = _geom_color(m, seg)
    base_rgb = rgba[..., :3]

    miss = (seg < 0).unsqueeze(-1)

    if background is not None:
        bg = torch.tensor(background, dtype=base_rgb.dtype, device=device)
    else:
        bg = torch.zeros(3, dtype=base_rgb.dtype, device=device)

    if shading and m.nlight > 0:
        hit_points = (origins_flat + dists.unsqueeze(-1) * dirs_flat).reshape(height, width, 3)
        normals = _compute_normals(
            hit_points.reshape(-1, 3),
            seg.reshape(-1),
            m,
            d,
        ).reshape(height, width, 3)
        rgb = _shade(
            normals.reshape(-1, 3),
            hit_points.reshape(-1, 3),
            dirs_flat,
            base_rgb.reshape(-1, 3),
            m,
            d,
        ).reshape(height, width, 3)
        rgb = torch.where(miss, bg, rgb)
    else:
        rgb = torch.where(miss, bg, base_rgb)

    return rgb, depth, seg
