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
"""Sensor functions."""

import mujoco
import torch

from mujoco_torch._src import math, ray, smooth
from mujoco_torch._src.types import Data, Model, ObjType, SensorType

_DATATYPE_REAL = int(mujoco.mjtDataType.mjDATATYPE_REAL)
_DATATYPE_POSITIVE = int(mujoco.mjtDataType.mjDATATYPE_POSITIVE)


def _apply_cutoff(sensor: torch.Tensor, cutoff: torch.Tensor, data_type: int) -> torch.Tensor:
    """Clip sensor to cutoff value."""

    def fn(sensor_elem, cutoff_elem):
        if data_type == _DATATYPE_REAL:
            return torch.where(cutoff_elem > 0, torch.clamp(sensor_elem, -cutoff_elem, cutoff_elem), sensor_elem)
        elif data_type == _DATATYPE_POSITIVE:
            return torch.where(cutoff_elem > 0, torch.minimum(sensor_elem, cutoff_elem), sensor_elem)
        else:
            return sensor_elem

    cutoff = torch.as_tensor(cutoff, dtype=sensor.dtype, device=sensor.device)
    return torch.vmap(fn)(sensor, cutoff)


def sensor_pos(m: Model, d: Data) -> Data:
    """Compute position-dependent sensors values."""
    if m.sensor_disabled_py:
        return d

    _dtype = d.qpos.dtype
    _dev = d.qpos.device
    objtype_data = {
        ObjType.UNKNOWN: (
            torch.zeros(1, 3, dtype=_dtype, device=_dev),
            torch.eye(3, dtype=_dtype, device=_dev).unsqueeze(0),
        ),
        ObjType.BODY: (d.xipos, d.ximat),
        ObjType.XBODY: (d.xpos, d.xmat),
        ObjType.GEOM: (d.geom_xpos, d.geom_xmat),
        ObjType.SITE: (d.site_xpos, d.site_xmat),
        ObjType.CAMERA: (d.cam_xpos, d.cam_xmat),
    }

    frame_axis = {
        SensorType.FRAMEXAXIS: 0,
        SensorType.FRAMEYAXIS: 1,
        SensorType.FRAMEZAXIS: 2,
    }

    sensors, adrs = [], []

    for group in m.sensor_groups_pos_py:
        sensor_type = group["type"]
        adr = group["adr"].to(_dev)
        cutoff = group["cutoff"]
        data_type = group["data_type"]
        objid = group["objid"]

        if sensor_type == SensorType.MAGNETOMETER:
            sensor = torch.vmap(lambda xmat: xmat.T @ m.opt.magnetic)(d.site_xmat[objid])
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.RANGEFINDER:
            for sub in group["body_groups"]:
                sub_objid = sub["objid"]
                site_xpos = d.site_xpos[sub_objid]
                site_mat = d.site_xmat[sub_objid].reshape((-1, 9))[:, torch.arange(3, device=d.qpos.device) * 3 + 2]
                sub_cutoff = sub["cutoff"]
                ray_precomp = sub["ray_precomp"]
                dist, _ = torch.vmap(
                    ray.ray_precomputed,
                    in_dims=(None, None, 0, 0),
                )(ray_precomp, d, site_xpos, site_mat)
                sensor = dist
                sensors.append(_apply_cutoff(sensor, sub_cutoff, data_type))
                adrs.append(sub["adr"].to(_dev))
            continue
        elif sensor_type == SensorType.JOINTPOS:
            sensor = d.qpos[group["qposadr"]]
        elif sensor_type == SensorType.TENDONPOS:
            sensor = d.ten_length[objid]
        elif sensor_type == SensorType.ACTUATORPOS:
            sensor = d.actuator_length[objid]
        elif sensor_type == SensorType.BALLQUAT:
            quat = d.qpos[group["qposadr_2d"]]
            sensor = torch.vmap(math.normalize)(quat)
            adr = (adr[:, None] + torch.arange(4, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.FRAMEPOS:

            def _framepos(xpos, xpos_ref, xmat_ref, refid):
                return torch.where(refid == -1, xpos, xmat_ref.T @ (xpos - xpos_ref))

            for sub in group["ot_rt_groups"]:
                ot, rt = sub["ot"], sub["rt"]
                sub_objid = sub["objid"]
                sub_refid = sub["refid"]
                xpos, _ = objtype_data[ot]
                xpos_ref, xmat_ref = objtype_data[rt]
                xpos = xpos[sub_objid]
                xpos_ref = xpos_ref[sub_refid]
                xmat_ref = xmat_ref[sub_refid]
                sub_cutoff = sub["cutoff"]
                sensor = torch.vmap(_framepos)(xpos, xpos_ref, xmat_ref, sub_refid)
                adrt = sub["adr"].to(_dev)[:, None] + torch.arange(3, device=_dev)
                sensors.append(_apply_cutoff(sensor, sub_cutoff, data_type).reshape(-1))
                adrs.append(adrt.reshape(-1))
            continue
        elif sensor_type in frame_axis:

            def _frameaxis(xmat, xmat_ref, refid):
                axis = xmat[:, frame_axis[sensor_type]]
                return torch.where(refid == -1, axis, xmat_ref.T @ axis)

            for sub in group["ot_rt_groups"]:
                ot, rt = sub["ot"], sub["rt"]
                sub_objid = sub["objid"]
                sub_refid = sub["refid"]
                _, xmat = objtype_data[ot]
                _, xmat_ref = objtype_data[rt]
                xmat = xmat[sub_objid]
                xmat_ref = xmat_ref[sub_refid]
                sub_cutoff = sub["cutoff"]
                sensor = torch.vmap(_frameaxis)(xmat, xmat_ref, sub_refid)
                adrt = sub["adr"].to(_dev)[:, None] + torch.arange(3, device=_dev)
                sensors.append(_apply_cutoff(sensor, sub_cutoff, data_type).reshape(-1))
                adrs.append(adrt.reshape(-1))
            continue
        elif sensor_type == SensorType.FRAMEQUAT:

            def _quat(otype, oid, bodyid):
                if otype == ObjType.XBODY:
                    return d.xquat[oid]
                elif otype == ObjType.BODY:
                    return torch.vmap(math.quat_mul)(d.xquat[oid], m.body_iquat[oid])
                elif otype == ObjType.GEOM:
                    return torch.vmap(math.quat_mul)(d.xquat[bodyid], m.geom_quat[oid])
                elif otype == ObjType.SITE:
                    return torch.vmap(math.quat_mul)(d.xquat[bodyid], m.site_quat[oid])
                elif otype == ObjType.CAMERA:
                    return torch.vmap(math.quat_mul)(d.xquat[bodyid], m.cam_quat[oid])
                elif otype == ObjType.UNKNOWN:
                    return torch.tile(torch.eye(4, dtype=d.qpos.dtype, device=d.qpos.device)[0], (oid.shape[0], 1))
                else:
                    raise ValueError(f"Unknown object type: {otype}")

            for sub in group["ot_rt_groups"]:
                ot, rt = sub["ot"], sub["rt"]
                sub_objid = sub["objid"]
                sub_refid = sub["refid"]
                quat = _quat(ot, sub_objid, sub["obj_bodyid"])
                refquat = _quat(rt, sub_refid, sub["ref_bodyid"])
                sub_cutoff = sub["cutoff"]
                sensor = torch.vmap(lambda q, r, rid: torch.where(rid == -1, q, math.quat_mul(math.quat_inv(r), q)))(
                    quat, refquat, sub_refid
                )
                adrt = sub["adr"].to(_dev)[:, None] + torch.arange(4, device=_dev)
                sensors.append(_apply_cutoff(sensor, sub_cutoff, data_type).reshape(-1))
                adrs.append(adrt.reshape(-1))
            continue
        elif sensor_type == SensorType.SUBTREECOM:
            sensor = d.subtree_com[objid]
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.CLOCK:
            sensor = torch.repeat_interleave(d.time, group["count"])
        else:
            continue

        sensors.append(_apply_cutoff(sensor, cutoff, data_type).reshape(-1))
        adrs.append(adr)

    if not adrs:
        return d

    adrs_flat = torch.cat(adrs)
    sensors_flat = torch.cat(sensors)
    sensordata = d.sensordata.clone()
    sensordata[adrs_flat.to(device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

    return d.replace(sensordata=sensordata)


def sensor_vel(m: Model, d: Data) -> Data:
    """Compute velocity-dependent sensors values."""
    if m.sensor_disabled_py:
        return d

    _dev = d.qpos.device
    groups = m.sensor_groups_vel_py
    group_types = {g["type"] for g in groups}

    if group_types & {SensorType.SUBTREELINVEL, SensorType.SUBTREEANGMOM}:
        if hasattr(smooth, "subtree_vel"):
            d = smooth.subtree_vel(m, d)

    sensors, adrs = [], []
    for group in groups:
        sensor_type = group["type"]
        adr = group["adr"].to(_dev)
        cutoff = group["cutoff"]
        data_type = group["data_type"]
        objid = group["objid"]

        if sensor_type == SensorType.VELOCIMETER:
            bodyid = group["bodyid"]
            pos = d.site_xpos[objid]
            rot = d.site_xmat[objid]
            cvel = d.cvel[bodyid]
            subtree_com = d.subtree_com[group["rootid"]]
            sensor = torch.vmap(lambda vec, dif, rot: rot.T @ (vec[3:] - math.cross(dif, vec[:3])))(
                cvel, pos - subtree_com, rot
            )
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.GYRO:
            bodyid = group["bodyid"]
            rot = d.site_xmat[objid]
            ang = d.cvel[bodyid, :3]
            sensor = torch.vmap(lambda ang, rot: rot.T @ ang)(ang, rot)
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.JOINTVEL:
            sensor = d.qvel[group["dofadr"]]
        elif sensor_type == SensorType.TENDONVEL:
            sensor = d.ten_velocity[objid]
        elif sensor_type == SensorType.ACTUATORVEL:
            sensor = d.actuator_velocity[objid]
        elif sensor_type == SensorType.BALLANGVEL:
            sensor = d.qvel[group["dofadr_2d"]]
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.SUBTREELINVEL:
            sensor = d.subtree_linvel[objid]
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.SUBTREEANGMOM:
            sensor = d.subtree_angmom[objid]
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        else:
            continue

        sensors.append(_apply_cutoff(sensor, cutoff, data_type).reshape(-1))
        adrs.append(adr)

    if not adrs:
        return d

    adrs_flat = torch.cat(adrs)
    sensors_flat = torch.cat(sensors)
    sensordata = d.sensordata.clone()
    sensordata[adrs_flat.to(device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

    return d.replace(sensordata=sensordata)


def sensor_acc(m: Model, d: Data) -> Data:
    """Compute acceleration/force-dependent sensors values."""
    if m.sensor_disabled_py:
        return d

    _dev = d.qpos.device
    groups = m.sensor_groups_acc_py
    group_types = {g["type"] for g in groups}

    if group_types & {
        SensorType.ACCELEROMETER,
        SensorType.FORCE,
        SensorType.TORQUE,
        SensorType.FRAMELINACC,
        SensorType.FRAMEANGACC,
    }:
        if hasattr(smooth, "rne_postconstraint"):
            d = smooth.rne_postconstraint(m, d)

    sensors, adrs = [], []
    for group in groups:
        sensor_type = group["type"]
        adr = group["adr"].to(_dev)
        cutoff = group["cutoff"]
        data_type = group["data_type"]
        objid = group["objid"]

        if sensor_type == SensorType.ACCELEROMETER:
            if hasattr(d, "cacc"):

                def _accelerometer(cvel, cacc, diff, rot):
                    ang = rot.T @ cvel[:3]
                    lin = rot.T @ (cvel[3:] - math.cross(diff, cvel[:3]))
                    acc = rot.T @ (cacc[3:] - math.cross(diff, cacc[:3]))
                    correction = math.cross(ang, lin)
                    return acc + correction

                bodyid = group["bodyid"]
                rot = d.site_xmat[objid]
                cvel = d.cvel[bodyid]
                cacc = d.cacc[bodyid]
                dif = d.site_xpos[objid] - d.subtree_com[group["rootid"]]

                sensor = torch.vmap(_accelerometer)(cvel, cacc, dif, rot)
            else:
                continue
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.FORCE:
            if hasattr(d, "cfrc_int"):
                bodyid = group["bodyid"]
                cfrc_int = d.cfrc_int[bodyid]
                site_xmat = d.site_xmat[objid]
                sensor = torch.vmap(lambda mat, vec: mat.T @ vec)(site_xmat, cfrc_int[:, 3:])
            else:
                continue
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.TORQUE:
            if hasattr(d, "cfrc_int"):
                bodyid = group["bodyid"]
                rootid = group["rootid"]
                cfrc_int = d.cfrc_int[bodyid]
                site_xmat = d.site_xmat[objid]
                dif = d.site_xpos[objid] - d.subtree_com[rootid]

                def _torque(vec, dif, rot):
                    return rot.T @ (vec[:3] - math.cross(dif, vec[3:]))

                sensor = torch.vmap(_torque)(cfrc_int, dif, site_xmat)
            else:
                continue
            adr = (adr[:, None] + torch.arange(3, device=_dev)).reshape(-1)
        elif sensor_type == SensorType.ACTUATORFRC:
            sensor = d.actuator_force[objid]
        elif sensor_type == SensorType.JOINTACTFRC:
            sensor = d.qfrc_actuator[group["dofadr"]]
        elif sensor_type == SensorType.TENDONACTFRC:
            force_mask = group["force_mask"].to(
                dtype=d.actuator_force.dtype,
                device=d.actuator_force.device,
            )
            sensor = force_mask @ d.actuator_force
        else:
            continue

        sensors.append(_apply_cutoff(sensor, cutoff, data_type).reshape(-1))
        adrs.append(adr)

    if not adrs:
        return d

    adrs_flat = torch.cat(adrs)
    sensors_flat = torch.cat(sensors)
    sensordata = d.sensordata.clone()
    sensordata[adrs_flat.to(device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

    return d.replace(sensordata=sensordata)
