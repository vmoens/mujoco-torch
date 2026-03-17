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

from mujoco_torch._src import math as mjmath
from mujoco_torch._src.ray import (
    _PRIMITIVE_RAY_FUNC,
    _ray_triangle,
    ray_precomputed,
)
from mujoco_torch._src.types import Data, GeomType, Model

# ---------------------------------------------------------------------------
# Pre-computation (model-level, done once)
# ---------------------------------------------------------------------------


def precompute_render_data(m: Model) -> dict:
    """Pre-compute scene geometry data for rendering.

    Call once per model and pass the result to :func:`render` via the
    ``precomp`` keyword to avoid repeated numpy work on every frame.

    Args:
      m: MJX Model.

    Returns:
      A dict with ``"prim"`` (primitive geom entries for
      :func:`ray_precomputed`), ``"geom_type_t"`` (geom type tensor),
      and optionally ``"mesh"`` (mesh triangle data).
    """
    geom_matid = np.asarray(m.geom_matid.cpu())
    geom_rgba = np.asarray(m.geom_rgba.cpu())
    geom_size = np.asarray(m.geom_size.cpu())
    geom_type = np.asarray(m.geom_type)

    mat_rgba = np.asarray(m.mat_rgba.cpu()) if m.mat_rgba.shape[0] > 0 else np.empty((0, 4))

    visible = (geom_matid != -1) | (geom_rgba[:, 3] != 0)
    if mat_rgba.shape[0] > 0:
        visible = visible & ((geom_matid == -1) | (mat_rgba[np.clip(geom_matid, 0, None), 3] != 0))

    prim_entries = []
    for gt, fn in _PRIMITIVE_RAY_FUNC.items():
        (id_np,) = np.nonzero(geom_type == gt)
        if id_np.size == 0:
            continue
        id_t = torch.tensor(id_np, dtype=torch.long)
        size_t = torch.as_tensor(geom_size[id_np].copy(), dtype=torch.float64)
        vis_t = torch.tensor(visible[id_np])
        prim_entries.append((fn, id_t, size_t, vis_t))

    result: dict = {
        "prim": tuple(prim_entries),
        "geom_type_t": torch.tensor(geom_type, dtype=torch.long),
    }

    # --- Mesh triangle data ---
    mesh_tri_verts_list: list[np.ndarray] = []
    mesh_tri_geom_ids_list: list[np.ndarray] = []

    mesh_geom_ids = np.nonzero(geom_type == int(GeomType.MESH))[0]
    if mesh_geom_ids.size > 0 and m.mesh_face.shape[0] > 0:
        faceadr = np.asarray(m.mesh_faceadr)
        vertadr = np.asarray(m.mesh_vertadr)
        mesh_face = np.asarray(m.mesh_face)
        mesh_vert = np.asarray(m.mesh_vert)
        n_meshes = faceadr.shape[0]
        nface_total = mesh_face.shape[0]
        nvert_total = mesh_vert.shape[0]

        for gid in mesh_geom_ids:
            if not visible[gid]:
                continue
            data_id = int(m.geom_dataid[gid])
            if data_id < 0 or data_id >= n_meshes:
                continue
            f_start = int(faceadr[data_id])
            f_end = int(faceadr[data_id + 1]) if data_id + 1 < n_meshes else nface_total
            v_start = int(vertadr[data_id])
            v_end = int(vertadr[data_id + 1]) if data_id + 1 < n_meshes else nvert_total
            faces = mesh_face[f_start:f_end]
            verts = mesh_vert[v_start:v_end]
            # faces index into the local vertex array; offset to global
            local_verts = verts[faces - faces.min()]  # (nface, 3, 3)
            # Actually, faces index into the mesh-local vertex range
            local_verts = verts[faces - v_start]
            mesh_tri_verts_list.append(local_verts.astype(np.float64))
            mesh_tri_geom_ids_list.append(np.full(faces.shape[0], gid, dtype=np.int64))

    if mesh_tri_verts_list:
        result["mesh_verts"] = torch.tensor(np.concatenate(mesh_tri_verts_list), dtype=torch.float64)
        result["mesh_geom_ids"] = torch.tensor(np.concatenate(mesh_tri_geom_ids_list), dtype=torch.long)

    # --- Texture data ---
    if m.ntex > 0:
        _precompute_textures(m, result)

    return result


def _precompute_textures(m: Model, result: dict) -> None:
    """Load texture data into tensors and store in *result*."""
    if m.ntex == 0 or m.ntexdata == 0:
        return

    tex_data = np.asarray(m.tex_data)
    tex_height = np.asarray(m.tex_height)
    tex_width = np.asarray(m.tex_width)
    tex_nchannel = np.asarray(m.tex_nchannel)
    tex_adr = np.asarray(m.tex_adr)

    # mat_texid is (nmat, 10) — column 1 is the 2D texture index
    mat_texid_np = np.asarray(m.mat_texid)
    if mat_texid_np.ndim == 2:
        mat_texid_2d = mat_texid_np[:, 1]
    else:
        mat_texid_2d = mat_texid_np

    textures: list[torch.Tensor] = []
    for i in range(m.ntex):
        h, w = int(tex_height[i]), int(tex_width[i])
        nc = int(tex_nchannel[i])
        start = int(tex_adr[i])
        n_pixels = h * w * nc
        pixels = tex_data[start : start + n_pixels].reshape(h, w, nc)
        if nc == 1:
            pixels = np.repeat(pixels, 3, axis=2)
        elif nc == 4:
            pixels = pixels[:, :, :3]
        textures.append(torch.tensor(pixels.astype(np.float64) / 255.0, dtype=torch.float64))

    result["textures"] = textures
    result["mat_texid_2d"] = torch.tensor(mat_texid_2d, dtype=torch.long)
    result["tex_height"] = torch.tensor(tex_height, dtype=torch.long)
    result["tex_width"] = torch.tensor(tex_width, dtype=torch.long)


def _resolve_precomp(precomp: dict, device: torch.device) -> dict:
    """Move precomp tensors to the target device."""
    resolved_prim = []
    for fn, id_t, size_t, vis_t in precomp["prim"]:
        resolved_prim.append((fn, id_t.to(device), size_t.to(device), vis_t.to(device)))
    out: dict = {
        "prim": tuple(resolved_prim),
        "geom_type_t": precomp["geom_type_t"].to(device),
    }
    if "mesh_verts" in precomp:
        out["mesh_verts"] = precomp["mesh_verts"].to(device)
        out["mesh_geom_ids"] = precomp["mesh_geom_ids"].to(device)
    if "textures" in precomp:
        out["textures"] = [t.to(device) for t in precomp["textures"]]
        out["mat_texid_2d"] = precomp["mat_texid_2d"].to(device)
        out["tex_height"] = precomp["tex_height"].to(device)
        out["tex_width"] = precomp["tex_width"].to(device)
    return out


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------


def _generate_rays(
    cam_xpos: torch.Tensor,
    cam_xmat: torch.Tensor,
    fovy_deg: float,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray origins and directions for every pixel.

    Uses the MuJoCo camera convention: -z forward, +x right, +y up.

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


# ---------------------------------------------------------------------------
# Colour / material lookup
# ---------------------------------------------------------------------------


def _geom_color(
    m: Model,
    geom_ids: torch.Tensor,
) -> torch.Tensor:
    """Look up RGBA colour for hit geom ids.

    Returns ``(..., 4)`` RGBA tensor.
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


# ---------------------------------------------------------------------------
# Texture sampling
# ---------------------------------------------------------------------------


def _compute_uv(
    hit_local: torch.Tensor,
    geom_size: torch.Tensor,
    geom_type_t: torch.Tensor,
) -> torch.Tensor:
    """Compute UV coordinates in [0,1] for hit points in local geom frame.

    Returns ``(N, 2)`` UV tensor.
    """
    # Plane: project xy onto size
    u_plane = hit_local[..., 0] / geom_size[..., 0].clamp(min=1e-10) * 0.5 + 0.5
    v_plane = hit_local[..., 1] / geom_size[..., 1].clamp(min=1e-10) * 0.5 + 0.5
    uv_plane = torch.stack([u_plane, v_plane], dim=-1)

    # Sphere / Ellipsoid: spherical mapping
    n = hit_local / geom_size.clamp(min=1e-10)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    u_sphere = torch.atan2(n[..., 1], n[..., 0]) / (2 * torch.pi) + 0.5
    v_sphere = torch.asin(n[..., 2].clamp(-1, 1)) / torch.pi + 0.5
    uv_sphere = torch.stack([u_sphere, v_sphere], dim=-1)

    # Cylinder / Capsule: cylindrical mapping
    u_cyl = torch.atan2(hit_local[..., 1], hit_local[..., 0]) / (2 * torch.pi) + 0.5
    half_len = geom_size[..., 1].clamp(min=1e-10)
    v_cyl = hit_local[..., 2] / (2 * half_len) + 0.5
    uv_cyl = torch.stack([u_cyl, v_cyl], dim=-1)

    # Box: use dominant face axis for planar projection
    abs_local = hit_local.abs()
    abs_scaled = abs_local / geom_size.clamp(min=1e-10)
    face_idx = abs_scaled.argmax(dim=-1)
    # face 0 (x-dominant): use y, z
    # face 1 (y-dominant): use x, z
    # face 2 (z-dominant): use x, y
    uv_map_idx = torch.tensor(
        [[1, 2], [0, 2], [0, 1]],
        dtype=torch.long,
        device=hit_local.device,
    )
    idx = uv_map_idx[face_idx]
    u_box = (
        hit_local.gather(-1, idx[..., 0:1]).squeeze(-1)
        / geom_size.gather(-1, idx[..., 0:1]).squeeze(-1).clamp(min=1e-10)
        * 0.5
        + 0.5
    )
    v_box = (
        hit_local.gather(-1, idx[..., 1:2]).squeeze(-1)
        / geom_size.gather(-1, idx[..., 1:2]).squeeze(-1).clamp(min=1e-10)
        * 0.5
        + 0.5
    )
    uv_box = torch.stack([u_box, v_box], dim=-1)

    # Select by geom type
    uv = uv_plane
    uv = torch.where((geom_type_t == int(GeomType.SPHERE)).unsqueeze(-1), uv_sphere, uv)
    uv = torch.where((geom_type_t == int(GeomType.ELLIPSOID)).unsqueeze(-1), uv_sphere, uv)
    uv = torch.where((geom_type_t == int(GeomType.CAPSULE)).unsqueeze(-1), uv_cyl, uv)
    uv = torch.where((geom_type_t == int(GeomType.CYLINDER)).unsqueeze(-1), uv_cyl, uv)
    uv = torch.where((geom_type_t == int(GeomType.BOX)).unsqueeze(-1), uv_box, uv)
    return uv.clamp(0, 1)


def _sample_texture(
    tex: torch.Tensor,
    uv: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sample from a single (H, W, 3) texture at (N, 2) UVs.

    Returns (N, 3).
    """
    h, w, _ = tex.shape
    u = uv[..., 0] * (w - 1)
    v = (1 - uv[..., 1]) * (h - 1)

    u0 = u.long().clamp(0, w - 2)
    v0 = v.long().clamp(0, h - 2)
    u1 = u0 + 1
    v1 = v0 + 1

    fu = (u - u0.float()).unsqueeze(-1)
    fv = (v - v0.float()).unsqueeze(-1)

    c00 = tex[v0, u0]
    c01 = tex[v0, u1]
    c10 = tex[v1, u0]
    c11 = tex[v1, u1]

    return c00 * (1 - fu) * (1 - fv) + c01 * fu * (1 - fv) + c10 * (1 - fu) * fv + c11 * fu * fv


def _apply_textures(
    base_rgb: torch.Tensor,
    hit_local: torch.Tensor,
    geom_size: torch.Tensor,
    geom_type_t: torch.Tensor,
    geom_ids: torch.Tensor,
    m: Model,
    precomp: dict,
) -> torch.Tensor:
    """Modulate base_rgb with texture colour where applicable.

    Returns ``(N, 3)`` RGB.
    """
    if "textures" not in precomp:
        return base_rgb

    mat_texid_2d = precomp["mat_texid_2d"]
    textures = precomp["textures"]

    safe_ids = geom_ids.clamp(min=0)
    mat_ids = m.geom_matid[safe_ids]
    has_mat = mat_ids >= 0
    tex_ids = mat_texid_2d[mat_ids.clamp(min=0)]
    has_tex = has_mat & (tex_ids >= 0)

    uv = _compute_uv(hit_local, geom_size, geom_type_t)

    tex_color = torch.ones_like(base_rgb)
    for tex_idx, tex_tensor in enumerate(textures):
        sampled = _sample_texture(tex_tensor, uv)
        mask = has_tex & (tex_ids == tex_idx)
        tex_color = torch.where(mask.unsqueeze(-1), sampled, tex_color)

    return base_rgb * tex_color


# ---------------------------------------------------------------------------
# Surface normals
# ---------------------------------------------------------------------------


def _safe_normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize vectors, returning zero for degenerate inputs."""
    return v / v.norm(dim=dim, keepdim=True).clamp(min=1e-10)


def _compute_normals(
    hit_points: torch.Tensor,
    geom_ids: torch.Tensor,
    m: Model,
    d: Data,
    geom_type_t: torch.Tensor,
) -> torch.Tensor:
    """Compute world-frame surface normals at hit points.

    Returns ``(N, 3)`` unit normals (zero for misses).
    """
    safe_ids = geom_ids.clamp(min=0)

    geom_xpos = d.geom_xpos[safe_ids]
    geom_xmat = d.geom_xmat[safe_ids]
    geom_size = m.geom_size[safe_ids]
    gt = geom_type_t[safe_ids]

    hit_local = (geom_xmat.transpose(-1, -2) @ (hit_points - geom_xpos).unsqueeze(-1)).squeeze(-1)

    # --- per-type normals in local frame ---

    # Plane: always (0, 0, 1)
    n_plane = torch.zeros_like(hit_local)
    n_plane[..., 2] = 1.0

    # Sphere
    n_sphere = _safe_normalize(hit_local)

    # Ellipsoid
    n_ellipsoid = _safe_normalize(hit_local / geom_size.square().clamp(min=1e-10))

    # Box
    abs_scaled = hit_local.abs() / geom_size.clamp(min=1e-10)
    face_idx = abs_scaled.argmax(dim=-1, keepdim=True)
    n_box = torch.zeros_like(hit_local)
    n_box.scatter_(-1, face_idx, torch.sign(hit_local).gather(-1, face_idx))

    # Capsule: cap vs cylinder side
    cap_z = torch.sign(hit_local[..., 2]) * geom_size[..., 1]
    cap_center = torch.zeros_like(hit_local)
    cap_center[..., 2] = cap_z
    n_cap = _safe_normalize(hit_local - cap_center)
    n_cyl_side = torch.zeros_like(hit_local)
    n_cyl_side[..., :2] = _safe_normalize(hit_local[..., :2], dim=-1)
    on_cap = hit_local[..., 2].abs() > geom_size[..., 1]
    n_capsule = torch.where(on_cap.unsqueeze(-1), n_cap, n_cyl_side)

    # Cylinder: flat caps vs round side
    n_cyl_round = torch.zeros_like(hit_local)
    n_cyl_round[..., :2] = _safe_normalize(hit_local[..., :2], dim=-1)
    n_cap_top = torch.zeros_like(hit_local)
    n_cap_top[..., 2] = torch.sign(hit_local[..., 2])
    on_cap_cyl = hit_local[..., 2].abs() > (geom_size[..., 1] - 1e-6)
    n_cylinder = torch.where(on_cap_cyl.unsqueeze(-1), n_cap_top, n_cyl_round)

    # --- select by geom type ---
    n_local = n_plane
    n_local = torch.where((gt == int(GeomType.SPHERE)).unsqueeze(-1), n_sphere, n_local)
    n_local = torch.where((gt == int(GeomType.CAPSULE)).unsqueeze(-1), n_capsule, n_local)
    n_local = torch.where(
        (gt == int(GeomType.ELLIPSOID)).unsqueeze(-1),
        n_ellipsoid,
        n_local,
    )
    n_local = torch.where((gt == int(GeomType.BOX)).unsqueeze(-1), n_box, n_local)
    n_local = torch.where(
        (gt == int(GeomType.CYLINDER)).unsqueeze(-1),
        n_cylinder,
        n_local,
    )
    # Mesh normals: approximate from face normal via cross product
    # (computed by the mesh intersection path and stored on the geom normal
    #  for now we fall back to sphere-like radial normal as approximation)
    n_local = torch.where(
        (gt == int(GeomType.MESH)).unsqueeze(-1),
        _safe_normalize(hit_local),
        n_local,
    )

    # transform to world frame
    n_world = (geom_xmat @ n_local.unsqueeze(-1)).squeeze(-1)

    miss = (geom_ids < 0).unsqueeze(-1)
    return torch.where(miss, torch.zeros_like(n_world), n_world)


# ---------------------------------------------------------------------------
# Shadow rays
# ---------------------------------------------------------------------------


def _shadow_test(
    hit_points: torch.Tensor,
    to_light: torch.Tensor,
    light_dist: torch.Tensor,
    precomp: dict,
    d: Data,
) -> torch.Tensor:
    """Test whether *hit_points* are in shadow for a given light.

    Casts a secondary ray from each hit point towards the light and returns
    a boolean mask that is *True* where the point is occluded.

    Args:
      hit_points: ``(N, 3)`` world-frame surface positions.
      to_light: ``(N, 3)`` normalised direction toward the light.
      light_dist: ``(N,)`` distance to the light (inf for directional).
      precomp: pre-computed render data (primitive entries).
      d: current Data.

    Returns:
      ``(N,)`` bool tensor — *True* means the sample is in shadow.
    """
    eps = 1e-4
    origins = hit_points + to_light * eps

    dists, _gids = torch.vmap(
        lambda p, v: ray_precomputed(precomp["prim"], d, p, v),
    )(origins, to_light)

    # In shadow when the secondary ray hits something closer than the light
    return (dists > 0) & (dists < light_dist - 2 * eps)


# ---------------------------------------------------------------------------
# Shading
# ---------------------------------------------------------------------------


def _shade(
    normals: torch.Tensor,
    hit_points: torch.Tensor,
    view_dirs: torch.Tensor,
    base_color: torch.Tensor,
    m: Model,
    d: Data,
    precomp: dict | None = None,
    shadows: bool = False,
) -> torch.Tensor:
    """Apply Lambert diffuse + Phong specular shading with attenuation.

    Returns ``(N, 3)`` shaded RGB clamped to [0, 1].
    """
    shaded = torch.zeros_like(base_color)

    for i in range(m.nlight):
        light_type_i = int(m.light_type[i])
        is_directional = light_type_i == 1

        # direction and distance from hit point towards the light
        to_light_dir = -d.light_xdir[i].expand_as(hit_points)
        to_light_pos = d.light_xpos[i] - hit_points
        to_light_raw = torch.where(
            torch.tensor(is_directional, device=hit_points.device).unsqueeze(-1),
            to_light_dir,
            to_light_pos,
        )
        light_dist_raw = to_light_raw.norm(dim=-1)
        to_light = _safe_normalize(to_light_raw)

        light_diff = m.light_diffuse[i]
        light_amb = m.light_ambient[i]
        light_spec = m.light_specular[i]

        # --- attenuation (Phase 6) ---
        atten = m.light_attenuation[i]
        ld = light_dist_raw
        att_factor = torch.where(
            torch.tensor(is_directional, device=hit_points.device),
            torch.ones_like(ld),
            1.0 / (atten[0] + atten[1] * ld + atten[2] * ld * ld).clamp(min=1e-10),
        )

        # --- spotlight cutoff (Phase 6) ---
        cutoff_deg = float(m.light_cutoff[i])
        spot_atten = torch.ones_like(ld)
        if cutoff_deg < 180.0:
            cos_cutoff = torch.cos(
                torch.tensor(
                    cutoff_deg * (torch.pi / 180.0),
                    dtype=ld.dtype,
                    device=ld.device,
                )
            )
            spot_dir = _safe_normalize(d.light_xdir[i]).expand_as(to_light)
            cos_angle = (-to_light * spot_dir).sum(dim=-1)
            spot_atten = torch.where(
                cos_angle > cos_cutoff,
                cos_angle.clamp(min=0).pow(10),
                torch.zeros_like(cos_angle),
            )

        # --- shadow test (Phase 5) ---
        shadow_mask = torch.zeros(
            hit_points.shape[0],
            dtype=torch.bool,
            device=hit_points.device,
        )
        if shadows and precomp is not None and bool(m.light_castshadow[i]):
            inf = torch.full_like(ld, torch.inf)
            shadow_dist = torch.where(
                torch.tensor(is_directional, device=hit_points.device),
                inf,
                light_dist_raw,
            )
            shadow_mask = _shadow_test(hit_points, to_light, shadow_dist, precomp, d)

        # --- Lambert diffuse ---
        ndotl = (normals * to_light).sum(dim=-1, keepdim=True).clamp(min=0)
        diffuse = base_color * light_diff * ndotl

        # --- Phong specular: R = 2(N·L)N - L ---
        reflect = 2 * ndotl * normals - to_light
        reflect = _safe_normalize(reflect)
        rdotv = (-view_dirs * reflect).sum(dim=-1, keepdim=True).clamp(min=0)
        specular = light_spec * rdotv.pow(50)

        ambient = base_color * light_amb

        # Combine: ambient is always added; diffuse+specular modulated
        light_contrib = ambient + (diffuse + specular) * att_factor.unsqueeze(-1) * spot_atten.unsqueeze(-1)

        # Zero out diffuse+specular for shadowed points (keep ambient)
        light_contrib = torch.where(
            shadow_mask.unsqueeze(-1),
            ambient,
            light_contrib,
        )

        shaded = shaded + light_contrib

    return shaded.clamp(0, 1)


# ---------------------------------------------------------------------------
# Mesh intersection (Phase 4b)
# ---------------------------------------------------------------------------


def _intersect_meshes(
    precomp: dict,
    d: Data,
    origins_flat: torch.Tensor,
    dirs_flat: torch.Tensor,
    prim_dists: torch.Tensor,
    prim_geom_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Test mesh triangles and merge with primitive intersection results.

    This iterates over mesh geoms in Python (unavoidable due to variable
    triangle counts) but vmaps the triangle tests per-mesh, per-pixel.

    Returns updated ``(dists, geom_ids)`` tensors.
    """
    mesh_verts = precomp["mesh_verts"]  # (N_tri, 3, 3)
    mesh_geom_ids = precomp["mesh_geom_ids"]  # (N_tri,)

    dists = prim_dists.clone()
    geom_ids = prim_geom_ids.clone()

    unique_gids = mesh_geom_ids.unique()
    for gid in unique_gids:
        gid_int = int(gid)
        tri_mask = mesh_geom_ids == gid
        tri_verts = mesh_verts[tri_mask]  # (n_tri, 3, 3)

        geom_xpos = d.geom_xpos[gid_int]
        geom_xmat = d.geom_xmat[gid_int]

        # transform rays to geom-local frame
        local_pnts = (geom_xmat.T @ (origins_flat - geom_xpos).unsqueeze(-1)).squeeze(-1)
        local_vecs = (geom_xmat.T @ dirs_flat.unsqueeze(-1)).squeeze(-1)

        # For each pixel, test all triangles of this mesh
        def _test_pixel(lpnt, lvec):
            b0, b1 = mjmath.orthogonals(mjmath.normalize(lvec))
            basis = torch.stack([b0, b1], dim=-1)
            tri_dists = torch.vmap(lambda v: _ray_triangle(v, lpnt, lvec, basis))(tri_verts)
            best = tri_dists.min()
            return best

        mesh_dists = torch.vmap(_test_pixel)(local_pnts, local_vecs)

        closer = (mesh_dists > 0) & ~torch.isinf(mesh_dists)
        closer = closer & ((dists < 0) | (mesh_dists < dists))
        dists = torch.where(closer, mesh_dists, dists)
        geom_ids = torch.where(
            closer,
            torch.full_like(geom_ids, gid_int),
            geom_ids,
        )

    return dists, geom_ids


# ---------------------------------------------------------------------------
# Fog (Phase 9)
# ---------------------------------------------------------------------------


def _apply_fog(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    fog_color: tuple[float, float, float],
    fog_start: float,
    fog_end: float,
) -> torch.Tensor:
    """Apply linear distance fog to the rendered image.

    Returns ``(H, W, 3)`` fogged RGB.
    """
    device = rgb.device
    dtype = rgb.dtype
    fc = torch.tensor(fog_color, dtype=dtype, device=device)
    d = depth.clamp(min=0)
    factor = ((d - fog_start) / (fog_end - fog_start)).clamp(0, 1)
    # Don't apply fog to missed pixels (depth < 0)
    factor = torch.where(depth < 0, torch.zeros_like(factor), factor)
    return rgb * (1 - factor.unsqueeze(-1)) + fc * factor.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Main render entry points
# ---------------------------------------------------------------------------


def render(
    m: Model,
    d: Data,
    camera_id: int = 0,
    width: int = 64,
    height: int = 64,
    precomp: dict | None = None,
    shading: bool = True,
    background: tuple[float, float, float] | None = None,
    shadows: bool = False,
    fog: tuple[tuple[float, float, float], float, float] | None = None,
    ssaa: int = 1,
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
      shadows: when *True*, cast secondary shadow rays toward each light
         source to detect occlusion.
      fog: optional ``(color, start, end)`` tuple for linear distance fog.
      ssaa: super-sample anti-aliasing factor.  Render at ``ssaa`` times
         the resolution and downsample.  ``1`` = no AA.

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

    # Super-sampling: render at higher resolution, downsample at the end
    render_w = width * ssaa
    render_h = height * ssaa

    cam_xpos = d.cam_xpos[camera_id]
    cam_xmat = d.cam_xmat[camera_id]
    fovy_deg = float(m.cam_fovy[camera_id])

    origins, directions = _generate_rays(cam_xpos, cam_xmat, fovy_deg, render_w, render_h)

    origins_flat = origins.reshape(-1, 3)
    dirs_flat = directions.reshape(-1, 3)

    dists, geom_ids = torch.vmap(
        lambda p, v: ray_precomputed(precomp["prim"], d, p, v),
    )(origins_flat, dirs_flat)

    # --- Mesh intersection (Phase 4b) ---
    if "mesh_verts" in precomp:
        dists, geom_ids = _intersect_meshes(precomp, d, origins_flat, dirs_flat, dists, geom_ids)

    depth = dists.reshape(render_h, render_w)
    seg = geom_ids.reshape(render_h, render_w)

    rgba = _geom_color(m, seg)
    base_rgb = rgba[..., :3]

    miss = (seg < 0).unsqueeze(-1)

    bg = (
        torch.tensor(background, dtype=base_rgb.dtype, device=device)
        if background is not None
        else torch.zeros(3, dtype=base_rgb.dtype, device=device)
    )

    # --- Texture modulation (Phase 8) ---
    if "textures" in precomp:
        safe_ids = seg.reshape(-1).clamp(min=0)
        geom_xpos_flat = d.geom_xpos[safe_ids]
        geom_xmat_flat = d.geom_xmat[safe_ids]
        geom_size_flat = m.geom_size[safe_ids]
        gt_flat = precomp["geom_type_t"][safe_ids]
        hp_flat = origins_flat + dists.unsqueeze(-1) * dirs_flat
        hl_flat = (geom_xmat_flat.transpose(-1, -2) @ (hp_flat - geom_xpos_flat).unsqueeze(-1)).squeeze(-1)
        base_rgb_flat = _apply_textures(
            base_rgb.reshape(-1, 3),
            hl_flat,
            geom_size_flat,
            gt_flat,
            seg.reshape(-1),
            m,
            precomp,
        )
        base_rgb = base_rgb_flat.reshape(render_h, render_w, 3)

    if shading and m.nlight > 0:
        hit_points = (origins_flat + dists.unsqueeze(-1) * dirs_flat).reshape(render_h, render_w, 3)
        normals = _compute_normals(
            hit_points.reshape(-1, 3),
            seg.reshape(-1),
            m,
            d,
            precomp["geom_type_t"],
        ).reshape(render_h, render_w, 3)
        rgb = _shade(
            normals.reshape(-1, 3),
            hit_points.reshape(-1, 3),
            dirs_flat,
            base_rgb.reshape(-1, 3),
            m,
            d,
            precomp=precomp if shadows else None,
            shadows=shadows,
        ).reshape(render_h, render_w, 3)
        rgb = torch.where(miss, bg, rgb)
    else:
        rgb = torch.where(miss, bg, base_rgb)

    # --- Fog (Phase 9) ---
    if fog is not None:
        fog_color, fog_start, fog_end = fog
        rgb = _apply_fog(rgb, depth, fog_color, fog_start, fog_end)

    # --- Downsample for SSAA ---
    if ssaa > 1:
        rgb = rgb.reshape(height, ssaa, width, ssaa, 3).mean(dim=(1, 3))
        depth = depth.reshape(height, ssaa, width, ssaa).mean(dim=(1, 3))
        # For segmentation, take the center sample
        seg = seg[ssaa // 2 :: ssaa, ssaa // 2 :: ssaa]

    return rgb, depth, seg


def render_batch(
    m: Model,
    d_batch: Data,
    camera_id: int = 0,
    width: int = 64,
    height: int = 64,
    precomp: dict | None = None,
    shading: bool = True,
    background: tuple[float, float, float] | None = None,
    shadows: bool = False,
    fog: tuple[tuple[float, float, float], float, float] | None = None,
    ssaa: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render all environments in a batch using ``torch.vmap``.

    Same as :func:`render` but accepts a **batched** ``Data`` (with a
    leading batch dimension) and returns batched outputs.

    Returns:
      ``(rgb, depth, seg)`` with an extra leading batch dimension:

      * **rgb**: ``(B, H, W, 3)``
      * **depth**: ``(B, H, W)``
      * **seg**: ``(B, H, W)``
    """
    if precomp is None:
        precomp = precompute_render_data(m)

    def _render_single(d):
        return render(
            m,
            d,
            camera_id=camera_id,
            width=width,
            height=height,
            precomp=precomp,
            shading=shading,
            background=background,
            shadows=shadows,
            fog=fog,
            ssaa=ssaa,
        )

    return torch.vmap(_render_single)(d_batch)
