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
"""Get and put mujoco data on/off device."""

import copy
import dataclasses
import warnings
from collections.abc import Iterable
from typing import Any, overload

# from torch import numpy as torch
import mujoco
import numpy as np
import torch
from torch.utils._pytree import tree_map

from mujoco_torch._src import collision_driver, mesh, ray, scan, types
from mujoco_torch._src.dataclasses import MjTensorClass

_MJ_TYPE_ATTR = {
    mujoco.mjtBias: (mujoco.MjModel.actuator_biastype,),
    mujoco.mjtDyn: (mujoco.MjModel.actuator_dyntype,),
    mujoco.mjtEq: (mujoco.MjModel.eq_type,),
    mujoco.mjtGain: (mujoco.MjModel.actuator_gaintype,),
    mujoco.mjtTrn: (mujoco.MjModel.actuator_trntype,),
    mujoco.mjtCone: (
        mujoco.MjModel.opt,
        mujoco.MjOption.cone,
    ),
    mujoco.mjtIntegrator: (
        mujoco.MjModel.opt,
        mujoco.MjOption.integrator,
    ),
    mujoco.mjtSolver: (
        mujoco.MjModel.opt,
        mujoco.MjOption.solver,
    ),
}

_TYPE_MAP = {
    mujoco._structs._MjContactList: types.Contact,  # pylint: disable=protected-access
    mujoco.MjData: types.Data,
    mujoco.MjModel: types.Model,
    mujoco.MjOption: types.Option,
    mujoco.MjStatistic: types.Statistic,
    mujoco.mjtBias: types.BiasType,
    mujoco.mjtCone: types.ConeType,
    mujoco.mjtDisableBit: types.DisableBit,
    mujoco.mjtDyn: types.DynType,
    mujoco.mjtEq: types.EqType,
    mujoco.mjtGain: types.GainType,
    mujoco.mjtIntegrator: types.IntegratorType,
    mujoco.mjtSolver: types.SolverType,
    mujoco.mjtTrn: types.TrnType,
}

_TRANSFORMS = {
    (types.Data, "ximat"): lambda x: x.reshape(x.shape[:-1] + (3, 3)),
    (types.Data, "xmat"): lambda x: x.reshape(x.shape[:-1] + (3, 3)),
    (types.Data, "geom_xmat"): lambda x: x.reshape(x.shape[:-1] + (3, 3)),
    (types.Data, "site_xmat"): lambda x: x.reshape(x.shape[:-1] + (3, 3)),
    (types.Data, "cam_xmat"): lambda x: x.reshape(x.shape[:-1] + (3, 3)),
    (types.Contact, "frame"): (
        lambda x: (
            x.reshape(x.shape[:-1] + (3, 3))  # pylint: disable=g-long-lambda
            if x is not None and x.shape[0]
            else torch.zeros((0, 3, 3))
        )
    ),
}

_INVERSE_TRANSFORMS = {
    (types.Data, "ximat"): lambda x: x.reshape(x.shape[:-2] + (9,)),
    (types.Data, "xmat"): lambda x: x.reshape(x.shape[:-2] + (9,)),
    (types.Data, "geom_xmat"): lambda x: x.reshape(x.shape[:-2] + (9,)),
    (types.Data, "site_xmat"): lambda x: x.reshape(x.shape[:-2] + (9,)),
    (types.Data, "cam_xmat"): lambda x: x.reshape(x.shape[:-2] + (9,)),
    # actuator_moment is (nu, nv) in torch, but (nu,) diagonal in MJ.
    # Extract the diagonal (nonzero per row) for MuJoCo format.
    (types.Data, "actuator_moment"): lambda x: x.amax(dim=-1) if x.ndim >= 2 else x,
    (types.Contact, "frame"): (
        lambda x: (
            x.reshape(x.shape[:-2] + (9,))  # pylint: disable=g-long-lambda
            if x is not None and x.shape[0]
            else torch.zeros((0, 9))
        )
    ),
}

# MuJoCo uses 'dim' for contact dimension; we use 'contact_dim' to avoid conflict with m.opt.dim
_FIELD_SOURCE_MAP = {(types.Contact, "contact_dim"): "dim"}  # device_put: read from Mujoco's 'dim'
_FIELD_TARGET_MAP = {(types.Contact, "contact_dim"): "dim"}  # device_get_into: write to Mujoco's 'dim'

_DERIVED = mesh.DERIVED.union(
    {
        # efc_J is dense in MJX, sparse in MJ. ignore for now.
        (types.Data, "efc_J"),
        # actuator_moment is (nu,) in MJ but (nu, nv) in torch
        (types.Data, "actuator_moment"),
        # qM, qLD have different sparse formats in MJ vs torch
        (types.Data, "qM"),
        (types.Data, "qLD"),
        (types.Option, "has_fluid_params"),
        # Torch-impl derived model fields
        (types.Model, "mesh_convex"),
        (types.Model, "dof_hasfrictionloss"),
        (types.Model, "geom_rbound_hfield"),
        (types.Model, "tendon_hasfrictionloss"),
        (types.Model, "has_gravcomp"),
        (types.Model, "dof_tri_row"),
        (types.Model, "dof_tri_col"),
        (types.Model, "actuator_info"),
        (types.Model, "constraint_sizes_py"),
        (types.Model, "condim_counts_py"),
        (types.Model, "condim_tensor_py"),
        (types.Model, "constraint_data_py"),
        (types.Model, "collision_groups_py"),
        (types.Model, "collision_max_cp_py"),
        (types.Model, "collision_total_contacts_py"),
        (types.Model, "sensor_groups_pos_py"),
        (types.Model, "sensor_groups_vel_py"),
        (types.Model, "sensor_groups_acc_py"),
        (types.Model, "sensor_disabled_py"),
        (types.Model, "cache_id"),
        # Pre-cached tensor versions of numpy model fields
        (types.Model, "body_rootid_t"),
        (types.Model, "dof_bodyid_t"),
        (types.Model, "dof_Madr_t"),
        (types.Model, "dof_tri_row_t"),
        (types.Model, "dof_tri_col_t"),
        (types.Model, "geom_bodyid_t"),
        (types.Model, "site_bodyid_t"),
        (types.Model, "dof_jntid_t"),
        (types.Model, "actuator_ctrllimited_bool"),
        (types.Model, "actuator_forcelimited_bool"),
        (types.Model, "jnt_actfrclimited_bool"),
        (types.Model, "actuator_actlimited_bool"),
        (types.Model, "actuator_actadr_neg1"),
        (types.Model, "sparse_i_t"),
        (types.Model, "sparse_j_t"),
        (types.Model, "sparse_madr_t"),
        (types.Model, "factor_m_madr_ds_t"),
        (types.Model, "factor_m_updates"),
        (types.Model, "solve_m_updates_j"),
        (types.Model, "solve_m_updates_i"),
    }
)


def _device_put_torch(x):
    if x is None:
        return
    try:
        # Python scalar floats: use float64 to match DEFAULT_DTYPE precision
        if isinstance(x, float):
            return torch.tensor(x, dtype=torch.float64)
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
        # Use torch.tensor() to copy data (torch.as_tensor shares memory
        # with numpy arrays, which can be mutated by MuJoCo in-place)
        return torch.tensor(x)
    except (RuntimeError, TypeError):
        return torch.empty(x.shape if isinstance(x, np.ndarray) else len(x))


torch.device_put = _device_put_torch


_model_cache_id_counter = 0


def _compute_condim_counts(value: mujoco.MjModel) -> tuple[int, int]:
    """Pre-compute per-condim contact counts (ncon_condim1, ncon_condim3)."""
    disableflags = int(value.opt.disableflags)
    if disableflags & (types.DisableBit.CONSTRAINT | types.DisableBit.CONTACT):
        return (0, 0)
    dims = collision_driver.make_condim(value)
    ncon_fl = int((dims == 1).sum())
    ncon_fr = int((dims == 3).sum())
    return (ncon_fl, ncon_fr)


def _compute_constraint_sizes(value: mujoco.MjModel) -> tuple[int, int, int, int, int]:
    """Pre-compute constraint sizes from raw MjModel (all numpy, no tracing)."""
    disableflags = int(value.opt.disableflags)
    if disableflags & types.DisableBit.CONSTRAINT:
        return (0, 0, 0, 0, 0)

    if disableflags & types.DisableBit.EQUALITY:
        ne = 0
    else:
        ne_connect = int((value.eq_type == types.EqType.CONNECT).sum())
        ne_weld = int((value.eq_type == types.EqType.WELD).sum())
        ne_joint = int((value.eq_type == types.EqType.JOINT).sum())
        ne = ne_connect * 3 + ne_weld * 6 + ne_joint

    if disableflags & types.DisableBit.FRICTIONLOSS:
        nf = 0
    else:
        nf = int((value.dof_frictionloss > 0).sum()) + int((value.tendon_frictionloss > 0).sum())

    if disableflags & types.DisableBit.LIMIT:
        nl = 0
    else:
        nl = int(value.jnt_limited.sum()) + int(value.tendon_limited.sum())

    if disableflags & types.DisableBit.CONTACT:
        ncon_ = 0
        nc = 0
    else:
        dims = collision_driver.make_condim(value)
        ncon_ = dims.numel()
        nc = int((dims == 1).sum()) + int((dims == 3).sum()) * 4

    nefc = ne + nf + nl + nc
    return (ne, nf, nl, ncon_, nefc)


def _compute_constraint_data(value: mujoco.MjModel) -> dict:
    """Pre-compute numpy-derived constraint data at device_put time.

    Returns a dict mapping constraint type names to their pre-computed
    data (or None if the type is inactive).  These are consumed by the
    ``_instantiate_*`` functions in ``constraint.py`` so that they never
    need ``np.nonzero`` or similar numpy ops at compile time.
    """
    disableflags = int(value.opt.disableflags)
    result: dict = {}

    if disableflags & types.DisableBit.CONSTRAINT:
        return {
            "eq_connect": None,
            "eq_weld": None,
            "eq_joint": None,
            "friction": None,
            "limit_ball": None,
            "limit_slide_hinge": None,
            "limit_tendon": None,
            "refsafe": bool(disableflags & types.DisableBit.REFSAFE),
        }

    # -- equality constraints --
    _cached = scan._cached_long
    if disableflags & types.DisableBit.EQUALITY:
        result["eq_connect"] = None
        result["eq_weld"] = None
        result["eq_joint"] = None
    else:
        for key, eq_val, multiplier in [
            ("eq_connect", types.EqType.CONNECT, 3),
            ("eq_weld", types.EqType.WELD, 6),
            ("eq_joint", types.EqType.JOINT, 1),
        ]:
            ids = np.nonzero(value.eq_type == eq_val)[0]
            if ids.size == 0:
                result[key] = None
                continue
            id1 = np.array(value.eq_obj1id[ids])
            id2 = np.array(value.eq_obj2id[ids])
            entry = {
                "ids": _cached(ids),
                "id1": _cached(id1),
                "id2": _cached(id2),
                "multiplier": multiplier,
            }
            if key == "eq_joint":
                entry["dofadr1"] = _cached(value.jnt_dofadr[id1])
                entry["dofadr2"] = _cached(value.jnt_dofadr[id2])
                entry["qposadr1"] = _cached(value.jnt_qposadr[id1])
                entry["qposadr2"] = _cached(value.jnt_qposadr[id2])
            result[key] = entry

    # -- friction --
    if disableflags & types.DisableBit.FRICTIONLOSS:
        result["friction"] = None
    else:
        dof_ids = np.nonzero(value.dof_frictionloss > 0)[0]
        tendon_ids = np.nonzero(value.tendon_frictionloss > 0)[0]
        size = int(dof_ids.size + tendon_ids.size)
        if size == 0:
            result["friction"] = None
        else:
            result["friction"] = {
                "dof_ids": _cached(dof_ids),
                "tendon_ids": _cached(tendon_ids),
                "size": size,
            }

    # -- limit: ball --
    if disableflags & types.DisableBit.LIMIT:
        result["limit_ball"] = None
        result["limit_slide_hinge"] = None
        result["limit_tendon"] = None
    else:
        ids = np.nonzero((value.jnt_type == types.JointType.BALL) & value.jnt_limited)[0]
        if ids.size == 0:
            result["limit_ball"] = None
        else:
            qposadr = torch.stack([torch.arange(q, q + 4) for q in value.jnt_qposadr[ids]])
            dofadr = torch.stack([torch.arange(da, da + 3) for da in value.jnt_dofadr[ids]])
            dofadr_first = torch.as_tensor(
                np.array(value.jnt_dofadr[ids]),
                dtype=torch.long,
            )
            result["limit_ball"] = {
                "ids": _cached(ids),
                "qposadr": scan._DeviceCachedTensor(qposadr),
                "dofadr": scan._DeviceCachedTensor(dofadr),
                "dofadr_first": scan._DeviceCachedTensor(dofadr_first),
            }

        slide_hinge = np.isin(value.jnt_type, (types.JointType.SLIDE, types.JointType.HINGE))
        ids = np.nonzero(slide_hinge & value.jnt_limited)[0]
        if ids.size == 0:
            result["limit_slide_hinge"] = None
        else:
            result["limit_slide_hinge"] = {
                "ids": _cached(ids),
                "qposadr": _cached(value.jnt_qposadr[ids]),
                "dofadr": _cached(value.jnt_dofadr[ids]),
            }

        tendon_id = np.nonzero(value.tendon_limited)[0]
        if tendon_id.size == 0:
            result["limit_tendon"] = None
        else:
            result["limit_tendon"] = {"tendon_id": _cached(tendon_id)}

    result["refsafe"] = bool(disableflags & types.DisableBit.REFSAFE)
    return result


def _compute_sensor_groups(value: mujoco.MjModel) -> dict[str, tuple]:
    """Pre-compute sensor group data for all three stages.

    Each stage produces a tuple of group dicts.  Every dict contains at
    least ``type``, ``data_type``, ``objid``, ``adr``, and ``cutoff``
    (all as tensors / ints).  Sensor-type-specific keys are added where
    the runtime computation would otherwise need numpy model indexing.
    """
    SType = types.SensorType
    OType = types.ObjType

    result: dict[str, tuple | bool] = {}
    result["sensor_disabled_py"] = bool(int(value.opt.disableflags) & types.DisableBit.SENSOR)

    for stage_key, stage_val in (
        ("sensor_groups_pos_py", mujoco.mjtStage.mjSTAGE_POS),
        ("sensor_groups_vel_py", mujoco.mjtStage.mjSTAGE_VEL),
        ("sensor_groups_acc_py", mujoco.mjtStage.mjSTAGE_ACC),
    ):
        stage_mask = value.sensor_needstage == stage_val
        sensor_type_values = sorted(set(value.sensor_type[stage_mask]))

        groups: list[dict] = []
        for st_val in sensor_type_values:
            idx = value.sensor_type == st_val
            objid_np = value.sensor_objid[idx]
            adr_np = value.sensor_adr[idx]
            cutoff_np = value.sensor_cutoff[idx]
            data_type_val = int(value.sensor_datatype[idx][0])

            group: dict = {
                "type": int(st_val),
                "data_type": data_type_val,
                "objid": torch.tensor(objid_np, dtype=torch.long),
                "adr": torch.tensor(adr_np, dtype=torch.long),
                "cutoff": torch.tensor(cutoff_np, dtype=torch.float64),
            }

            st_int = int(st_val)

            # -- position stage extras --
            if st_int == SType.JOINTPOS:
                group["qposadr"] = torch.tensor(
                    value.jnt_qposadr[objid_np],
                    dtype=torch.long,
                )
            elif st_int == SType.BALLQUAT:
                group["qposadr_2d"] = torch.tensor(
                    value.jnt_qposadr[objid_np, None] + np.arange(4)[None],
                    dtype=torch.long,
                )
            elif st_int == SType.RANGEFINDER:
                site_bodyid = value.site_bodyid[objid_np]
                body_groups: list[dict] = []
                for sid in sorted(set(site_bodyid)):
                    idxs = site_bodyid == sid
                    body_groups.append(
                        {
                            "sid": int(sid),
                            "objid": torch.tensor(objid_np[idxs], dtype=torch.long),
                            "cutoff": torch.tensor(cutoff_np[idxs], dtype=torch.float64),
                            "adr": torch.tensor(adr_np[idxs], dtype=torch.long),
                            "ray_precomp": ray.precompute_ray_data(
                                value,
                                flg_static=True,
                                bodyexclude=int(sid),
                            ),
                        }
                    )
                group["body_groups"] = tuple(body_groups)
            elif st_int in (
                SType.FRAMEPOS,
                SType.FRAMEXAXIS,
                SType.FRAMEYAXIS,
                SType.FRAMEZAXIS,
                SType.FRAMEQUAT,
            ):
                objtype_np = value.sensor_objtype[idx]
                reftype_np = value.sensor_reftype[idx]
                refid_np = value.sensor_refid[idx]

                _bodyid_src = {
                    int(OType.GEOM): value.geom_bodyid,
                    int(OType.SITE): value.site_bodyid,
                    int(OType.CAMERA): value.cam_bodyid,
                }

                ot_rt_groups: list[dict] = []
                for ot_val, rt_val in sorted(set(zip(objtype_np, reftype_np))):
                    idxt = (objtype_np == ot_val) & (reftype_np == rt_val)
                    sub_objid = objid_np[idxt]
                    sub_refid = refid_np[idxt]

                    sub: dict = {
                        "ot": int(ot_val),
                        "rt": int(rt_val),
                        "objid": torch.tensor(sub_objid, dtype=torch.long),
                        "refid": torch.tensor(sub_refid, dtype=torch.long),
                        "cutoff": torch.tensor(cutoff_np[idxt], dtype=torch.float64),
                        "adr": torch.tensor(adr_np[idxt], dtype=torch.long),
                        "obj_bodyid": None,
                        "ref_bodyid": None,
                    }

                    if st_int == SType.FRAMEQUAT:
                        if int(ot_val) in _bodyid_src:
                            sub["obj_bodyid"] = torch.tensor(
                                _bodyid_src[int(ot_val)][sub_objid],
                                dtype=torch.long,
                            )
                        if int(rt_val) in _bodyid_src:
                            sub["ref_bodyid"] = torch.tensor(
                                _bodyid_src[int(rt_val)][sub_refid],
                                dtype=torch.long,
                            )

                    ot_rt_groups.append(sub)
                group["ot_rt_groups"] = tuple(ot_rt_groups)
            elif st_int == SType.CLOCK:
                group["count"] = int(idx.sum())

            # -- velocity stage extras --
            elif st_int in (SType.VELOCIMETER, SType.GYRO):
                bodyid_np = value.site_bodyid[objid_np]
                group["bodyid"] = torch.tensor(bodyid_np, dtype=torch.long)
                group["rootid"] = torch.tensor(
                    value.body_rootid[bodyid_np],
                    dtype=torch.long,
                )
            elif st_int == SType.JOINTVEL:
                group["dofadr"] = torch.tensor(
                    value.jnt_dofadr[objid_np],
                    dtype=torch.long,
                )
            elif st_int == SType.BALLANGVEL:
                group["dofadr_2d"] = torch.tensor(
                    value.jnt_dofadr[objid_np, None] + np.arange(3)[None],
                    dtype=torch.long,
                )

            # -- acceleration stage extras --
            elif st_int in (SType.ACCELEROMETER, SType.FORCE, SType.TORQUE):
                bodyid_np = value.site_bodyid[objid_np]
                group["bodyid"] = torch.tensor(bodyid_np, dtype=torch.long)
                group["rootid"] = torch.tensor(
                    value.body_rootid[bodyid_np],
                    dtype=torch.long,
                )
            elif st_int == SType.JOINTACTFRC:
                group["dofadr"] = torch.tensor(
                    value.jnt_dofadr[objid_np],
                    dtype=torch.long,
                )
            elif st_int == SType.TENDONACTFRC:
                force_mask = np.stack(
                    [
                        (value.actuator_trntype == int(types.TrnType.TENDON)) & (value.actuator_trnid[:, 0] == tid)
                        for tid in objid_np
                    ]
                )
                group["force_mask"] = torch.tensor(force_mask, dtype=torch.float64)

            groups.append(group)

        result[stage_key] = tuple(groups)

    return result


def _model_derived(value: mujoco.MjModel) -> dict[str, Any]:
    global _model_cache_id_counter
    _model_cache_id_counter += 1

    mesh_kwargs = mesh.get(value)
    result = {"cache_id": _model_cache_id_counter}
    for k, v in mesh_kwargs.items():
        result[k] = tuple(torch.tensor(x) if x is not None else None for x in v)
    # mesh_convex: one ConvexMesh per mesh
    result["mesh_convex"] = tuple(mesh.convex(value, i) for i in range(value.nmesh))
    result["dof_hasfrictionloss"] = np.array(value.dof_frictionloss > 0)
    result["geom_rbound_hfield"] = np.array(value.geom_rbound)
    result["tendon_hasfrictionloss"] = np.array(value.tendon_frictionloss > 0)
    result["has_gravcomp"] = bool((value.body_gravcomp != 0).any())

    ij = []
    for i in range(value.nv):
        j = i
        while j > -1:
            ij.append((i, j))
            j = value.dof_parentid[j]
    rows, cols = zip(*ij) if ij else ((), ())
    result["dof_tri_row"] = np.array(rows, dtype=np.int64)
    result["dof_tri_col"] = np.array(cols, dtype=np.int64)

    actuator_info = []
    for i in range(value.nu):
        trntype = int(value.actuator_trntype[i])
        trnid = int(value.actuator_trnid[i, 0])
        jnt_type = dofadr = qposadr = 0
        if trntype != types.TrnType.TENDON:
            jnt_type = int(value.jnt_type[trnid])
            dofadr = int(value.jnt_dofadr[trnid])
            qposadr = int(value.jnt_qposadr[trnid])
        actuator_info.append((trntype, trnid, jnt_type, dofadr, qposadr))
    result["actuator_info"] = tuple(actuator_info)

    result["constraint_sizes_py"] = _compute_constraint_sizes(value)
    result["condim_counts_py"] = _compute_condim_counts(value)
    result["condim_tensor_py"] = collision_driver.make_condim(value)
    result["constraint_data_py"] = _compute_constraint_data(value)

    geom_convex_data = (
        result["geom_convex_face"],
        result["geom_convex_vert"],
        result["geom_convex_edge"],
    )
    candidate_set = collision_driver.collision_candidates(
        value,
        geom_convex_data=geom_convex_data,
    )
    max_cp = collision_driver._max_contact_points(value)
    collision_groups = []
    total_contacts = 0
    for key, cands in candidate_set.items():
        geom_types = (types.GeomType(int(key[0])), types.GeomType(int(key[1])))
        fn = collision_driver.get_collision_fn(geom_types)
        precomp = None
        if fn is not None and geom_types[0] != types.GeomType.HFIELD:
            precomp = collision_driver.precompute_collision_indices(cands)
        collision_groups.append((fn, geom_types, cands, precomp))
        if fn is not None:
            total_contacts += fn.ncon * len(cands)
    result["collision_groups_py"] = tuple(collision_groups)
    result["collision_max_cp_py"] = max_cp
    result["collision_total_contacts_py"] = total_contacts

    result.update(_compute_sensor_groups(value))

    scan.precompute_scan_caches(value, _model_cache_id_counter)

    # Pre-cached tensor versions of numpy model fields.
    # These are regular tensors on CPU; .to(device) on the Model moves them.
    result["body_rootid_t"] = torch.as_tensor(np.array(value.body_rootid), dtype=torch.long)
    result["dof_bodyid_t"] = torch.as_tensor(np.array(value.dof_bodyid), dtype=torch.long)
    result["dof_Madr_t"] = torch.as_tensor(np.array(value.dof_Madr), dtype=torch.long)
    result["dof_tri_row_t"] = torch.as_tensor(result["dof_tri_row"], dtype=torch.long)
    result["dof_tri_col_t"] = torch.as_tensor(result["dof_tri_col"], dtype=torch.long)
    result["geom_bodyid_t"] = torch.as_tensor(np.array(value.geom_bodyid), dtype=torch.long)
    result["site_bodyid_t"] = torch.as_tensor(np.array(value.site_bodyid), dtype=torch.long)
    result["dof_jntid_t"] = torch.as_tensor(np.array(value.dof_jntid), dtype=torch.long)
    result["actuator_ctrllimited_bool"] = (
        torch.as_tensor(np.array(value.actuator_ctrllimited)[:, None], dtype=torch.bool)
        if value.nu > 0
        else torch.empty((0, 1), dtype=torch.bool)
    )
    result["actuator_forcelimited_bool"] = (
        torch.as_tensor(np.array(value.actuator_forcelimited)[:, None], dtype=torch.bool)
        if value.nu > 0
        else torch.empty((0, 1), dtype=torch.bool)
    )
    result["jnt_actfrclimited_bool"] = (
        torch.as_tensor(np.array(value.jnt_actfrclimited)[:, None], dtype=torch.bool)
        if value.njnt > 0
        else torch.empty((0, 1), dtype=torch.bool)
    )
    result["actuator_actlimited_bool"] = (
        torch.as_tensor(np.array(value.actuator_actlimited)[:, None], dtype=torch.bool)
        if value.nu > 0
        else torch.empty((0, 1), dtype=torch.bool)
    )
    result["actuator_actadr_neg1"] = (
        torch.tensor(np.array(value.actuator_actadr) == -1, dtype=torch.bool)
        if value.nu > 0
        else torch.empty(0, dtype=torch.bool)
    )

    # Pre-compute sparse mass matrix index pattern
    is_, js, madr_ijs = [], [], []
    for i in range(value.nv):
        madr_ij, j = int(value.dof_Madr[i]), i
        while True:
            madr_ij, j = madr_ij + 1, int(value.dof_parentid[j])
            if j == -1:
                break
            is_.append(i)
            js.append(j)
            madr_ijs.append(madr_ij)
    result["sparse_i_t"] = torch.tensor(is_, dtype=torch.long) if is_ else torch.empty(0, dtype=torch.long)
    result["sparse_j_t"] = torch.tensor(js, dtype=torch.long) if js else torch.empty(0, dtype=torch.long)
    result["sparse_madr_t"] = torch.tensor(madr_ijs, dtype=torch.long) if madr_ijs else torch.empty(0, dtype=torch.long)

    # Pre-compute factor_m indices
    depth = []
    for i in range(value.nv):
        pid = int(value.dof_parentid[i])
        depth.append(depth[pid] + 1 if pid != -1 else 0)
    factor_updates = {}
    factor_madr_ds = []
    for i in range(value.nv):
        madr_d = madr_ij = int(value.dof_Madr[i])
        j = i
        while True:
            factor_madr_ds.append(madr_d)
            pid = int(value.dof_parentid[j])
            madr_ij, j = madr_ij + 1, pid
            if j == -1:
                break
            out_beg, out_end = int(value.dof_Madr[j]), int(value.dof_Madr[j + 1])
            factor_updates.setdefault(depth[j], []).append((out_beg, out_end, madr_d, madr_ij))
    result["factor_m_madr_ds_t"] = (
        torch.tensor(factor_madr_ds, dtype=torch.long) if factor_madr_ds else torch.empty(0, dtype=torch.long)
    )
    # Pre-compute factored update index tensors as _DeviceCachedTensor for lazy GPU transfer
    factor_m_updates_precomp = []
    for _, updates_list in sorted(factor_updates.items(), reverse=True):
        rows, madr_ijs_f, pivots, out = [], [], [], []
        for b, e, madr_d, madr_ij_v in updates_list:
            width = e - b
            rows.extend(range(madr_ij_v, madr_ij_v + width))
            madr_ijs_f.extend([madr_ij_v] * width)
            pivots.extend([madr_d] * width)
            out.extend(range(b, e))
        factor_m_updates_precomp.append(
            (
                scan._DeviceCachedTensor(torch.tensor(rows, dtype=torch.long)),
                scan._DeviceCachedTensor(torch.tensor(madr_ijs_f, dtype=torch.long)),
                scan._DeviceCachedTensor(torch.tensor(pivots, dtype=torch.long)),
                scan._DeviceCachedTensor(torch.tensor(out, dtype=torch.long)),
            )
        )
    result["factor_m_updates"] = tuple(factor_m_updates_precomp)

    # Pre-compute solve_m indices
    solve_updates_i, solve_updates_j = {}, {}
    for i in range(value.nv):
        madr_ij, j = int(value.dof_Madr[i]), i
        while True:
            pid = int(value.dof_parentid[j])
            madr_ij, j = madr_ij + 1, pid
            if j == -1:
                break
            solve_updates_i.setdefault(depth[i], []).append((i, madr_ij, j))
            solve_updates_j.setdefault(depth[j], []).append((j, madr_ij, i))

    def _build_solve_groups(updates_dict, reverse=False):
        groups = []
        for _, vals in sorted(updates_dict.items(), reverse=reverse):
            t = torch.tensor(vals, dtype=torch.long)
            groups.append(
                (
                    scan._DeviceCachedTensor(t[:, 0]),
                    scan._DeviceCachedTensor(t[:, 1]),
                    scan._DeviceCachedTensor(t[:, 2]),
                )
            )
        return tuple(groups)

    result["solve_m_updates_j"] = _build_solve_groups(solve_updates_j, reverse=True)
    result["solve_m_updates_i"] = _build_solve_groups(solve_updates_i, reverse=False)

    return result


def _data_derived(value: mujoco.MjData) -> dict[str, Any]:
    nv = value.qvel.shape[0]
    nu = value.ctrl.shape[0]
    # MuJoCo stores actuator_moment as (nu,), we need (nu, nv) dense
    actuator_moment = torch.zeros((nu, nv), dtype=torch.float64)
    if nu > 0:
        m_data = np.array(value.actuator_moment)
        actuator_moment = torch.as_tensor(
            np.diag(m_data) if nu == nv else np.zeros((nu, nv)),
            dtype=torch.float64,
        )
    return {
        "efc_J": torch.device_put(value.efc_J),
        "actuator_moment": actuator_moment,
        "qM": torch.as_tensor(value.qM.copy(), dtype=torch.float64),
        "qLD": torch.as_tensor(value.qM.copy(), dtype=torch.float64),  # same sparse format as qM
    }


def _option_derived(value: types.Option) -> dict[str, Any]:
    has_fluid = bool(value.density > 0 or value.viscosity > 0 or (value.wind != 0.0).any())
    return {"has_fluid_params": has_fluid}


def _validate(m: mujoco.MjModel):
    """Validates that an mjModel is compatible with MJX."""

    # check enum types
    for mj_type, attrs in _MJ_TYPE_ATTR.items():
        val = m
        for attr in attrs:
            val = attr.fget(val)  # pytype: disable=attribute-error

        typs = set(val) if isinstance(val, Iterable) else {val}
        unsupported_typs = typs - set(_TYPE_MAP[mj_type])
        unsupported = [mj_type(t) for t in unsupported_typs]  # pylint: disable=too-many-function-args
        if unsupported:
            raise NotImplementedError(f"{unsupported} not implemented.")

    # check condim
    if any(dim not in (1, 3) for dim in m.geom_condim) or any(dim not in (1, 3) for dim in m.pair_dim):
        raise NotImplementedError("Only condim=1 and condim=3 are supported.")

    # check collision geom types
    candidate_set = collision_driver.collision_candidates(m)
    for g1, g2, *_ in candidate_set:
        g1, g2 = mujoco.mjtGeom(g1), mujoco.mjtGeom(g2)
        if g1 == mujoco.mjtGeom.mjGEOM_PLANE and g2 in (
            mujoco.mjtGeom.mjGEOM_PLANE,
            mujoco.mjtGeom.mjGEOM_HFIELD,
        ):
            # MuJoCo does not collide planes with other planes or hfields
            continue
        if collision_driver.get_collision_fn((g1, g2)) is None:
            raise NotImplementedError(f"({g1}, {g2}) collisions not implemented.")

    # TODO(erikfrey): warn for high solver iterations, nefc, etc.

    # mjNDISABLE is not a DisableBit flag, so must be explicitly ignored
    disablebit_members = set(mujoco.mjtDisableBit.__members__.values()) - {mujoco.mjtDisableBit.mjNDISABLE}
    unsupported_disable = disablebit_members - {mujoco.mjtDisableBit(t.value) for t in types.DisableBit}
    for f in unsupported_disable:
        if f & m.opt.disableflags:
            warnings.warn(f"Ignoring disable flag {f.name}.")

    # mjNENABLE is not an EnableBit flag, so must be explicitly ignored
    unsupported_enable = set(mujoco.mjtEnableBit.__members__.values()) - {mujoco.mjtEnableBit.mjNENABLE}
    for f in unsupported_enable:
        if f & m.opt.enableflags:
            warnings.warn(f"Ignoring enable flag {f.name}.")


@overload
def device_put(value: mujoco.MjData, *, dtype: torch.dtype | None = None) -> types.Data: ...


@overload
def device_put(value: mujoco.MjModel, *, dtype: torch.dtype | None = None) -> types.Model: ...


def _cast_float(v, dtype):
    """Cast a tensor to *dtype* if it is floating-point; leave others unchanged."""
    if isinstance(v, torch.Tensor) and v.is_floating_point():
        return v.to(dtype)
    return v


def device_put(value, *, dtype: torch.dtype | None = None):
    """Places mujoco data onto a device.

    Args:
      value: a mujoco struct to transfer
      dtype: optional floating-point dtype override.  When set, every
        floating-point tensor in the output is cast to *dtype* (e.g.
        ``torch.float32``).  Integer and boolean tensors are unaffected.

    Returns:
      on-device MJX struct reflecting the input value
    """
    clz = _TYPE_MAP.get(type(value))
    if clz is None:
        raise NotImplementedError(f"{type(value)} is not supported for device_put.")

    if isinstance(value, mujoco.MjModel):
        _validate(value)  # type: ignore

    init_kwargs = {}
    for f in dataclasses.fields(clz):  # type: ignore
        if (clz, f.name) in _DERIVED:
            continue

        source_name = _FIELD_SOURCE_MAP.get((clz, f.name), f.name)
        field_value = getattr(value, source_name)
        if (clz, f.name) in _TRANSFORMS:
            field_value = _TRANSFORMS[(clz, f.name)](field_value)

        if f.type is torch.Tensor:
            field_value = torch.device_put(field_value)
        elif type(field_value) in _TYPE_MAP.keys():
            field_value = device_put(field_value, dtype=dtype)

        init_kwargs[f.name] = copy.copy(field_value)

    derived_kwargs = {}
    if isinstance(value, mujoco.MjModel):
        derived_kwargs = _model_derived(value)
    elif isinstance(value, mujoco.MjData):
        derived_kwargs = _data_derived(value)
    elif isinstance(value, mujoco.MjOption):
        derived_kwargs = _option_derived(value)

    if dtype is not None:
        for k, v in init_kwargs.items():
            init_kwargs[k] = _cast_float(v, dtype)
        for k, v in derived_kwargs.items():
            derived_kwargs[k] = _cast_float(v, dtype)

    if issubclass(clz, MjTensorClass):
        init_kwargs["batch_size"] = []
    result = clz(**init_kwargs, **derived_kwargs)  # type: ignore

    if clz is types.Model:
        from mujoco_torch._src.scan import _resolve_cached_tensors
        types._build_device_precomp(result, torch.device("cpu"), _resolve_cached_tensors)

    return result


@overload
def device_get_into(result: mujoco.MjData | list[mujoco.MjData], value: types.Data): ...


def device_get_into(result, value):
    """Transfers data off device into a mujoco MjData.

    Data on device often has a batch dimension which adds (N,) to the beginning
    of each array shape where N = batch size.

    If result is a single MjData, arrays are copied over with the batch dimension
    intact. If result is a list, the list must be length N and will be populated
    with distinct MjData structs where the batch dimension is stripped.

    Args:
      result: struct (or list of structs) to transfer into
      value: device value to transfer

    Raises:
      RuntimeError: if result length doesn't match data batch size
    """

    # In PyTorch, tensors are already accessible from CPU; no device_get needed.
    # Just ensure all tensors are detached and on CPU.
    value = tree_map(
        lambda x: x.detach().cpu() if isinstance(x, torch.Tensor) else x,
        value,
    )

    if isinstance(result, list):
        array_shapes = [s.shape for s in torch.utils._pytree.tree_flatten(value)[0]]

        if any(len(s) < 1 or s[0] != array_shapes[0][0] for s in array_shapes):
            raise ValueError("unrecognizable batch dimension in value")

        batch_size = array_shapes[0][0]

        if len(result) != batch_size:
            raise ValueError(f"result length ({len(result)}) doesn't match value batch size ({batch_size})")

        for i in range(batch_size):
            value_i = torch.utils._pytree.tree_map(lambda x, i=i: x[i], value)
            device_get_into(result[i], value_i)

    else:
        if isinstance(result, mujoco.MjData):
            mujoco._functions._realloc_con_efc(  # pylint: disable=protected-access
                result, ncon=int(value.ncon), nefc=int(value.nefc)
            )

        for f in dataclasses.fields(value):  # type: ignore
            if (type(value), f.name) in _DERIVED:
                continue

            field_value = getattr(value, f.name)

            if (type(value), f.name) in _INVERSE_TRANSFORMS:
                field_value = _INVERSE_TRANSFORMS[(type(value), f.name)](field_value)

            result_name = _FIELD_TARGET_MAP.get((type(value), f.name), f.name)

            if type(field_value) in _TYPE_MAP.values():
                device_get_into(getattr(result, result_name), field_value)
                continue

            # Convert torch tensors to numpy for MuJoCo compatibility
            if isinstance(field_value, torch.Tensor):
                field_value = field_value.detach().cpu().numpy()

            # Skip fields with incompatible shapes (e.g. different sparse formats)
            result_field = getattr(result, result_name, None)
            if (
                result_field is not None
                and hasattr(result_field, "shape")
                and hasattr(field_value, "shape")
                and result_field.shape != field_value.shape
            ):
                continue

            try:
                setattr(result, result_name, field_value)
            except (AttributeError, ValueError):
                getattr(result, result_name)[:] = field_value
