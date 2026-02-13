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
import numpy as np
import torch
from mujoco_torch._src import math
from mujoco_torch._src import ray
from mujoco_torch._src import smooth
from mujoco_torch._src import support
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import Model
from mujoco_torch._src.types import ObjType
from mujoco_torch._src.types import SensorType
from mujoco_torch._src.types import TrnType


def _apply_cutoff(
    sensor: torch.Tensor, cutoff: torch.Tensor, data_type: int
) -> torch.Tensor:
  """Clip sensor to cutoff value."""

  def fn(sensor_elem, cutoff_elem):
    if data_type == mujoco.mjtDataType.mjDATATYPE_REAL:
      return torch.where(cutoff_elem > 0, torch.clamp(sensor_elem, -cutoff_elem, cutoff_elem), sensor_elem)
    elif data_type == mujoco.mjtDataType.mjDATATYPE_POSITIVE:
      return torch.where(cutoff_elem > 0, torch.minimum(sensor_elem, cutoff_elem), sensor_elem)
    else:
      return sensor_elem

  cutoff = torch.as_tensor(cutoff, dtype=sensor.dtype, device=sensor.device) if not isinstance(cutoff, torch.Tensor) else cutoff
  return torch.vmap(fn)(sensor, cutoff)


@torch.compiler.disable
def sensor_pos(m: Model, d: Data) -> Data:
  """Compute position-dependent sensors values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  # position and orientation by object type
  objtype_data = {
      ObjType.UNKNOWN: (
          np.zeros((1, 3)),
          np.expand_dims(np.eye(3), axis=0),
      ),
      ObjType.BODY: (d.xipos, d.ximat),
      ObjType.XBODY: (d.xpos, d.xmat),
      ObjType.GEOM: (d.geom_xpos, d.geom_xmat),
      ObjType.SITE: (d.site_xpos, d.site_xmat),
      ObjType.CAMERA: (d.cam_xpos, d.cam_xmat),
  }

  # frame axis indexing
  frame_axis = {
      SensorType.FRAMEXAXIS: 0,
      SensorType.FRAMEYAXIS: 1,
      SensorType.FRAMEZAXIS: 2,
  }

  stage_pos = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_POS
  sensors, adrs = [], []

  for sensor_type in set(m.sensor_type[stage_pos]):
    idx = m.sensor_type == sensor_type
    objid = m.sensor_objid[idx]
    objtype = m.sensor_objtype[idx]
    refid = m.sensor_refid[idx]
    reftype = m.sensor_reftype[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.MAGNETOMETER:
      sensor = torch.vmap(lambda xmat: xmat.T @ m.opt.magnetic)(d.site_xmat[objid])
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.RANGEFINDER:
      site_bodyid = m.site_bodyid[objid]
      for sid in set(site_bodyid):
        idxs = sid == site_bodyid
        objids = objid[idxs]
        site_xpos = d.site_xpos[objids]
        site_mat = d.site_xmat[objids].reshape((-1, 9))[:, np.array([2, 5, 8])]
        cutoffs = cutoff[idxs]
        dist, _ = torch.vmap(
            ray.ray, in_dims=(None, None, 0, 0, None, None, None)
        )(m, d, site_xpos, site_mat, (), True, sid)
        sensor = dist
        sensors.append(_apply_cutoff(sensor, cutoffs, data_type[0]))
        adrs.append(adr[idxs])
      continue
    elif sensor_type == SensorType.JOINTPOS:
      sensor = d.qpos[torch.tensor(m.jnt_qposadr[objid], device=d.qpos.device)]
    elif sensor_type == SensorType.TENDONPOS:
      sensor = d.ten_length[objid]
    elif sensor_type == SensorType.ACTUATORPOS:
      sensor = d.actuator_length[objid]
    elif sensor_type == SensorType.BALLQUAT:
      jnt_qposadr = m.jnt_qposadr[objid, None] + np.arange(4)[None]
      quat = d.qpos[jnt_qposadr]
      sensor = torch.vmap(math.normalize)(quat)
      adr = (adr[:, None] + np.arange(4)[None]).reshape(-1)
    elif sensor_type == SensorType.FRAMEPOS:

      def _framepos(xpos, xpos_ref, xmat_ref, refid):
        return torch.where(refid == -1, xpos, xmat_ref.T @ (xpos - xpos_ref))

      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        refidt = refid[idxt]
        xpos, _ = objtype_data[ot]
        xpos_ref, xmat_ref = objtype_data[rt]
        xpos = xpos[objid[idxt]]
        xpos_ref = xpos_ref[refidt]
        xmat_ref = xmat_ref[refidt]
        cutofft = cutoff[idxt]
        sensor = torch.vmap(_framepos)(xpos, xpos_ref, xmat_ref, refidt)
        adrt = adr[idxt, None] + np.arange(3)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue
    elif sensor_type in frame_axis:

      def _frameaxis(xmat, xmat_ref, refid):
        axis = xmat[:, frame_axis[sensor_type]]
        return torch.where(refid == -1, axis, xmat_ref.T @ axis)

      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        refidt = refid[idxt]
        _, xmat = objtype_data[ot]
        _, xmat_ref = objtype_data[rt]
        xmat = xmat[objid[idxt]]
        xmat_ref = xmat_ref[refidt]
        cutofft = cutoff[idxt]
        sensor = torch.vmap(_frameaxis)(xmat, xmat_ref, refidt)
        adrt = adr[idxt, None] + np.arange(3)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue
    elif sensor_type == SensorType.FRAMEQUAT:

      def _quat(otype, oid):
        if otype == ObjType.XBODY:
          return d.xquat[oid]
        elif otype == ObjType.BODY:
          return torch.vmap(math.quat_mul)(d.xquat[oid], m.body_iquat[oid])
        elif otype == ObjType.GEOM:
          return torch.vmap(math.quat_mul)(
              d.xquat[m.geom_bodyid[oid]], m.geom_quat[oid]
          )
        elif otype == ObjType.SITE:
          return torch.vmap(math.quat_mul)(
              d.xquat[m.site_bodyid[oid]], m.site_quat[oid]
          )
        elif otype == ObjType.CAMERA:
          return torch.vmap(math.quat_mul)(
              d.xquat[m.cam_bodyid[oid]], m.cam_quat[oid]
          )
        elif otype == ObjType.UNKNOWN:
          return torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0], device=d.qpos.device), (oid.size, 1))
        else:
          raise ValueError(f'Unknown object type: {otype}')

      for ot, rt in set(zip(objtype, reftype)):
        idxt = (objtype == ot) & (reftype == rt)
        objidt = objid[idxt]
        refidt = refid[idxt]
        quat = _quat(ot, objidt)
        refquat = _quat(rt, refidt)
        cutofft = cutoff[idxt]
        sensor = torch.vmap(
            lambda q, r, rid: torch.where(
                rid == -1, q, math.quat_mul(math.quat_inv(r), q))
        )(quat, refquat, refidt)
        adrt = adr[idxt, None] + np.arange(4)[None]
        sensors.append(_apply_cutoff(sensor, cutofft, data_type[0]).reshape(-1))
        adrs.append(adrt.reshape(-1))
      continue
    elif sensor_type == SensorType.SUBTREECOM:
      sensor = d.subtree_com[objid]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.CLOCK:
      sensor = torch.repeat_interleave(d.time, idx.sum())
    else:
      continue

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))
    adrs.append(adr)

  if not adrs:
    return d

  adrs_flat = np.concatenate(adrs)
  sensors_flat = torch.cat(sensors)
  sensordata = d.sensordata.clone()
  sensordata[torch.tensor(adrs_flat, device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

  return d.replace(sensordata=sensordata)


@torch.compiler.disable
def sensor_vel(m: Model, d: Data) -> Data:
  """Compute velocity-dependent sensors values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  objtype_data = {
      ObjType.UNKNOWN: (
          np.zeros((1, 3)),
          np.expand_dims(np.eye(3), axis=0),
          np.arange(1),
      ),
      ObjType.BODY: (d.xipos, d.ximat, np.arange(m.nbody)),
      ObjType.XBODY: (d.xpos, d.xmat, np.arange(m.nbody)),
      ObjType.GEOM: (d.geom_xpos, d.geom_xmat, m.geom_bodyid),
      ObjType.SITE: (d.site_xpos, d.site_xmat, m.site_bodyid),
      ObjType.CAMERA: (d.cam_xpos, d.cam_xmat, m.cam_bodyid),
  }

  stage_vel = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_VEL
  sensor_types = set(m.sensor_type[stage_vel])

  if sensor_types & {SensorType.SUBTREELINVEL, SensorType.SUBTREEANGMOM}:
    if hasattr(smooth, 'subtree_vel'):
      d = smooth.subtree_vel(m, d)

  sensors, adrs = [], []
  for sensor_type in sensor_types:
    idx = m.sensor_type == sensor_type
    objid = m.sensor_objid[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.VELOCIMETER:
      bodyid = m.site_bodyid[objid]
      pos = d.site_xpos[objid]
      rot = d.site_xmat[objid]
      cvel = d.cvel[bodyid]
      subtree_com = d.subtree_com[m.body_rootid[bodyid]]
      sensor = torch.vmap(
          lambda vec, dif, rot: rot.T @ (vec[3:] - torch.linalg.cross(dif, vec[:3]))
      )(cvel, pos - subtree_com, rot)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.GYRO:
      bodyid = m.site_bodyid[objid]
      rot = d.site_xmat[objid]
      ang = d.cvel[bodyid, :3]
      sensor = torch.vmap(lambda ang, rot: rot.T @ ang)(ang, rot)
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.JOINTVEL:
      sensor = d.qvel[torch.tensor(m.jnt_dofadr[objid], device=d.qvel.device)]
    elif sensor_type == SensorType.TENDONVEL:
      sensor = d.ten_velocity[objid]
    elif sensor_type == SensorType.ACTUATORVEL:
      sensor = d.actuator_velocity[objid]
    elif sensor_type == SensorType.BALLANGVEL:
      jnt_dotadr = m.jnt_dofadr[objid, None] + np.arange(3)[None]
      sensor = d.qvel[jnt_dotadr]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.SUBTREELINVEL:
      sensor = d.subtree_linvel[objid]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.SUBTREEANGMOM:
      sensor = d.subtree_angmom[objid]
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    else:
      continue

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))
    adrs.append(adr)

  if not adrs:
    return d

  adrs_flat = np.concatenate(adrs)
  sensors_flat = torch.cat(sensors)
  sensordata = d.sensordata.clone()
  sensordata[torch.tensor(adrs_flat, device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

  return d.replace(sensordata=sensordata)


@torch.compiler.disable
def sensor_acc(m: Model, d: Data) -> Data:
  """Compute acceleration/force-dependent sensors values."""
  if m.opt.disableflags & DisableBit.SENSOR:
    return d

  objtype_data = {
      ObjType.UNKNOWN: (np.zeros((1, 3)), np.arange(1)),
      ObjType.BODY: (d.xipos, np.arange(m.nbody)),
      ObjType.XBODY: (d.xpos, np.arange(m.nbody)),
      ObjType.GEOM: (d.geom_xpos, m.geom_bodyid),
      ObjType.SITE: (d.site_xpos, m.site_bodyid),
      ObjType.CAMERA: (d.cam_xpos, m.cam_bodyid),
  }

  stage_acc = m.sensor_needstage == mujoco.mjtStage.mjSTAGE_ACC
  sensor_types = set(m.sensor_type[stage_acc])

  if sensor_types & {
      SensorType.ACCELEROMETER,
      SensorType.FORCE,
      SensorType.TORQUE,
      SensorType.FRAMELINACC,
      SensorType.FRAMEANGACC,
  }:
    if hasattr(smooth, 'rne_postconstraint'):
      d = smooth.rne_postconstraint(m, d)

  sensors, adrs = [], []
  for sensor_type in sensor_types:
    idx = m.sensor_type == sensor_type
    objid = m.sensor_objid[idx]
    adr = m.sensor_adr[idx]
    cutoff = m.sensor_cutoff[idx]
    data_type = m.sensor_datatype[idx]

    if sensor_type == SensorType.ACCELEROMETER:
      if hasattr(d, 'cacc'):

        def _accelerometer(cvel, cacc, diff, rot):
          ang = rot.T @ cvel[:3]
          lin = rot.T @ (cvel[3:] - torch.linalg.cross(diff, cvel[:3]))
          acc = rot.T @ (cacc[3:] - torch.linalg.cross(diff, cacc[:3]))
          correction = torch.linalg.cross(ang, lin)
          return acc + correction

        bodyid = m.site_bodyid[objid]
        rot = d.site_xmat[objid]
        cvel = d.cvel[bodyid]
        cacc = d.cacc[bodyid]
        dif = d.site_xpos[objid] - d.subtree_com[m.body_rootid[bodyid]]

        sensor = torch.vmap(_accelerometer)(cvel, cacc, dif, rot)
      else:
        continue
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.FORCE:
      if hasattr(d, 'cfrc_int'):
        bodyid = m.site_bodyid[objid]
        cfrc_int = d.cfrc_int[bodyid]
        site_xmat = d.site_xmat[objid]
        sensor = torch.vmap(lambda mat, vec: mat.T @ vec)(site_xmat, cfrc_int[:, 3:])
      else:
        continue
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.TORQUE:
      if hasattr(d, 'cfrc_int'):
        bodyid = m.site_bodyid[objid]
        rootid = m.body_rootid[bodyid]
        cfrc_int = d.cfrc_int[bodyid]
        site_xmat = d.site_xmat[objid]
        dif = d.site_xpos[objid] - d.subtree_com[rootid]

        def _torque(vec, dif, rot):
          return rot.T @ (vec[:3] - torch.linalg.cross(dif, vec[3:]))

        sensor = torch.vmap(_torque)(cfrc_int, dif, site_xmat)
      else:
        continue
      adr = (adr[:, None] + np.arange(3)[None]).reshape(-1)
    elif sensor_type == SensorType.ACTUATORFRC:
      sensor = d.actuator_force[objid]
    elif sensor_type == SensorType.JOINTACTFRC:
      sensor = d.qfrc_actuator[torch.tensor(m.jnt_dofadr[objid], device=d.qpos.device)]
    elif sensor_type == SensorType.TENDONACTFRC:
      force_mask = np.array([
          (m.actuator_trntype == TrnType.TENDON)
          & (m.actuator_trnid[:, 0] == tendon_id)
          for tendon_id in objid
      ], dtype=np.float32)
      force_mask_t = torch.tensor(force_mask, dtype=d.actuator_force.dtype, device=d.actuator_force.device)
      sensor = force_mask_t @ d.actuator_force
    else:
      continue

    sensors.append(_apply_cutoff(sensor, cutoff, data_type[0]).reshape(-1))
    adrs.append(adr)

  if not adrs:
    return d

  adrs_flat = np.concatenate(adrs)
  sensors_flat = torch.cat(sensors)
  sensordata = d.sensordata.clone()
  sensordata[torch.tensor(adrs_flat, device=sensordata.device)] = sensors_flat.to(sensordata.dtype)

  return d.replace(sensordata=sensordata)
