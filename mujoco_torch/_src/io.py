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
"""Functions to initialize, load, or save data."""

import copy
from typing import List, Union

import mujoco
from tensordict import TensorDict

from mujoco_torch._src.types import NonVmappableTensor
from mujoco_torch._src import collision_driver
from mujoco_torch._src import constraint
from mujoco_torch._src import mesh
from mujoco_torch._src import support
from mujoco_torch._src import types
import numpy as np
import torch
import scipy


def _put_option(o: mujoco.MjOption, device=None) -> types.Option:
  """Puts mujoco.MjOption onto a device, resulting in mujoco_torch.Option."""
  if o.integrator not in set(types.IntegratorType):
    raise NotImplementedError(f'{mujoco.mjtIntegrator(o.integrator)}')

  if o.cone not in set(types.ConeType):
    raise NotImplementedError(f'{mujoco.mjtCone(o.cone)}')

  if o.jacobian not in set(types.JacobianType):
    raise NotImplementedError(f'{mujoco.mjtJacobian(o.jacobian)}')

  if o.solver not in set(types.SolverType):
    raise NotImplementedError(f'{mujoco.mjtSolver(o.solver)}')

  for i in range(mujoco.mjtEnableBit.mjNENABLE):
    if o.enableflags & 2**i:
      raise NotImplementedError(f'{mujoco.mjtEnableBit(2 ** i)}')

  static_fields = TensorDict({
      f.name: getattr(o, f.name)
      for f in types.Option.fields()
      if f.type in (int, bytes, NonVmappableTensor)
  })
  static_fields['integrator'] = types.IntegratorType(o.integrator)
  static_fields['cone'] = types.ConeType(o.cone)
  static_fields['jacobian'] = types.JacobianType(o.jacobian)
  static_fields['solver'] = types.SolverType(o.solver)
  static_fields['disableflags'] = types.DisableBit(o.disableflags)

  device_fields = TensorDict({
      f.name: getattr(o, f.name)
      for f in types.Option.fields()
      if f.type is torch.Tensor
  })
  device_fields = device_fields.to(device=device)

  has_fluid_params = o.density > 0 or o.viscosity > 0 or o.wind.any()

  return types.Option(
      has_fluid_params=has_fluid_params,
      **static_fields,
      **device_fields,
  )


def _put_statistic(s: mujoco.MjStatistic, device=None) -> types.Statistic:
  """Puts mujoco.MjStatistic onto a device, resulting in mujoco_torch.Statistic."""
  return types.Statistic(
      meaninertia=s.meaninertia, device=device
  )


def put_model(m: mujoco.MjModel, device=None) -> types.Model:
  """Puts mujoco.MjModel onto a device, resulting in mujoco_torch.Model."""

  if m.ntendon:
    raise NotImplementedError('tendons are not supported')

  if (m.geom_condim != 3).any() or (m.pair_dim != 3).any():
    raise NotImplementedError('only condim=3 is supported')

  # check collision geom types
  for (g1, g2, *_), c in collision_driver.collision_candidates(m).items():
    g1, g2 = mujoco.mjtGeom(g1), mujoco.mjtGeom(g2)
    if collision_driver.get_collision_fn((g1, g2)) is None:
      raise NotImplementedError(f'({g1}, {g2}) has no collision function')
    *_, params = collision_driver.get_params(m, c)
    margin_gap = not np.allclose(np.concatenate([params.margin, params.gap]), 0)
    if mujoco.mjtGeom.mjGEOM_MESH in (g1, g2) and margin_gap:
      raise NotImplementedError(
          f'Margin and gap not implemented for ({g1}, {g2})'
      )

  for enum_field, enum_type, mj_type in (
      (m.actuator_biastype, types.BiasType, mujoco.mjtBias),
      (m.actuator_dyntype, types.DynType, mujoco.mjtDyn),
      (m.actuator_gaintype, types.GainType, mujoco.mjtGain),
      (m.actuator_trntype, types.TrnType, mujoco.mjtTrn),
      (m.eq_type, types.EqType, mujoco.mjtEq),
  ):
    missing = set(enum_field) - set(enum_type)
    if missing:
      raise NotImplementedError(
          f'{[mj_type(m) for m in missing]} not supported'
      )

  if not np.allclose(m.dof_frictionloss, 0):
    raise NotImplementedError('dof_frictionloss is not implemented.')

  opt = _put_option(m.opt, device=device)
  stat = _put_statistic(m.stat, device=device)

  def _replace_key(key):
      if key == "_names":
          return "names"
      return key

  def build_static_fiels():
      for f in types.Model.fields():
        if f.type in (int, bytes, NonVmappableTensor):
          val = getattr(m, _replace_key(f.name), None)
          if val is not None:
            yield f.name, val

  static_fields = TensorDict(dict(build_static_fiels()))
  static_fields['geom_rgba'] = static_fields['geom_rgba'].reshape((-1, 4))
  static_fields['mat_rgba'] = static_fields['mat_rgba'].reshape((-1, 4))

  device_fields = TensorDict({
      f.name: getattr(m, f.name)
      for f in types.Model.fields()
      if f.type is torch.Tensor
  })
  device_fields['cam_mat0'] = device_fields['cam_mat0'].reshape((-1, 3, 3))
  mesh_result = mesh.get(m)
  device_fields.update(mesh_result)
  device_fields = device_fields.to(device=device)

  return types.Model(
      opt=opt,
      stat=stat,
      **static_fields,
      **device_fields,
  )


def make_data(m: Union[types.Model, mujoco.MjModel]) -> types.Data:
  """Allocate and initialize Data."""

  ncon = collision_driver.ncon(m)
  ne, nf, nl, nc = constraint.count_constraints(m)
  nefc = ne + nf + nl + nc

  #TODO: no need to instantiate all of that here
  zero_0 = torch.zeros(0, dtype=torch.get_default_dtype())
  zero_nv = torch.zeros(m.nv, dtype=torch.get_default_dtype())
  zero_nv_6 = torch.zeros((m.nv, 6), dtype=torch.get_default_dtype())
  zero_nv_nv = torch.zeros((m.nv, m.nv), dtype=torch.get_default_dtype())
  zero_nbody_3 = torch.zeros((m.nbody, 3), dtype=torch.get_default_dtype())
  zero_nbody_6 = torch.zeros((m.nbody, 6), dtype=torch.get_default_dtype())
  zero_nbody_10 = torch.zeros((m.nbody, 10), dtype=torch.get_default_dtype())
  zero_nbody_3_3 = torch.zeros((m.nbody, 3, 3), dtype=torch.get_default_dtype())
  zero_nefc = torch.zeros(nefc, dtype=torch.get_default_dtype())
  zero_na = torch.zeros(m.na, dtype=torch.get_default_dtype())
  zero_nu = torch.zeros(m.nu, dtype=torch.get_default_dtype())
  zero_njnt_3 = torch.zeros((m.njnt, 3), dtype=torch.get_default_dtype())
  zero_nm = torch.zeros(m.nM, dtype=torch.get_default_dtype())

  # create first d to get num contacts and nc
  d = types.Data(
      solver_niter=torch.tensor(0, dtype=int),
      time=torch.tensor(0.0),
      qpos=torch.tensor(m.qpos0),
      qvel=zero_nv,
      act=zero_na,
      qacc_warmstart=zero_nv,
      ctrl=zero_nu,
      qfrc_applied=zero_nv,
      xfrc_applied=zero_nbody_6,
      eq_active=torch.zeros(m.neq, dtype=torch.uint8),
      qacc=zero_nv,
      act_dot=zero_na,
      xpos=zero_nbody_3,
      xquat=torch.zeros((m.nbody, 4), dtype=torch.get_default_dtype()),
      xmat=zero_nbody_3_3,
      xipos=zero_nbody_3,
      ximat=zero_nbody_3_3,
      xanchor=zero_njnt_3,
      xaxis=zero_njnt_3,
      geom_xpos=torch.zeros((m.ngeom, 3), dtype=torch.get_default_dtype()),
      geom_xmat=torch.zeros((m.ngeom, 3, 3), dtype=torch.get_default_dtype()),
      site_xpos=torch.zeros((m.nsite, 3), dtype=torch.get_default_dtype()),
      site_xmat=torch.zeros((m.nsite, 3, 3), dtype=torch.get_default_dtype()),
      cam_xpos=torch.zeros((m.ncam, 3), dtype=torch.get_default_dtype()),
      cam_xmat=torch.zeros((m.ncam, 3, 3), dtype=torch.get_default_dtype()),
      subtree_com=zero_nbody_3,
      cdof=zero_nv_6,
      cinert=zero_nbody_10,
      actuator_length=zero_nu,
      actuator_moment=torch.zeros((m.nu, m.nv), dtype=torch.get_default_dtype()),
      crb=zero_nbody_10,
      qM=zero_nm if support.is_sparse(m) else zero_nv_nv,
      qLD=zero_nm if support.is_sparse(m) else zero_nv_nv,
      qLDiagInv=zero_nv if support.is_sparse(m) else zero_0,
      contact=types.Contact.zero(ncon),
      efc_J=torch.zeros((nefc, m.nv), dtype=torch.get_default_dtype()),
      efc_frictionloss=zero_nefc,
      efc_D=zero_nefc,
      actuator_velocity=zero_nu,
      cvel=zero_nbody_6,
      cdof_dot=zero_nv_6,
      qfrc_bias=zero_nv,
      qfrc_passive=zero_nv,
      efc_aref=zero_nefc,
      qfrc_actuator=zero_nv,
      qfrc_smooth=zero_nv,
      qacc_smooth=zero_nv,
      qfrc_constraint=zero_nv,
      qfrc_inverse=zero_nv,
      efc_force=zero_nefc,
      userdata=torch.zeros(m.nuserdata, dtype=torch.get_default_dtype()),
  )

  return d


def _get_contact(
    c: mujoco._structs._MjContactList,
    cx: types.Contact,
    efc_start: int,
):
  """Converts mujoco_torch.Contact to mujoco._structs._MjContactList."""
  con_id = torch.nonzero(cx.dist <= 0)[0]
  for field in types.Contact.fields():
    value = getattr(cx, field.name)[con_id]
    if field.name == 'frame':
      value = value.reshape((-1, 9))
    getattr(c, field.name)[:] = value

  ncon = cx.dist.shape[0]
  c.efc_address[:] = np.arange(efc_start, efc_start + ncon * 4, 4)[con_id]
  c.dim[:] = 3


def get_data(
    m: mujoco.MjModel, d: types.Data
) -> Union[mujoco.MjData, List[mujoco.MjData]]:
  """Gets mujoco_torch.Data from a device, resulting in mujoco.MjData or List[MjData]."""
  batched = len(d.qpos.shape) > 1
  batch_size = d.qpos.shape[0] if batched else 1

  if batched:
    result = [mujoco.MjData(m) for _ in range(batch_size)]
  else:
    result = mujoco.MjData(m)

  get_data_into(result, m, d)

  return result


def get_data_into(
    result: Union[mujoco.MjData, List[mujoco.MjData]],
    m: mujoco.MjModel,
    d: types.Data,
):
  """Gets mujoco_torch.Data from a device into an existing mujoco.MjData or list."""
  batched = isinstance(result, list)
  if batched and len(d.qpos.shape) < 2:
    raise ValueError('dst is a list, but d is not batched.')
  if not batched and len(d.qpos.shape) >= 2:
    raise ValueError('dst is a an MjData, but d is batched.')

  d = jax.device_get(d)

  batch_size = d.qpos.shape[0] if batched else 1
  ne, nf, nl, nc = constraint.count_constraints(m, d)
  efc_type = np.array([
      mujoco.mjtConstraint.mjCNSTR_EQUALITY,
      mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
      mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,
      mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL,
  ]).repeat([ne, nf, nl, nc])

  dof_i, dof_j = [], []
  for i in range(m.nv):
    j = i
    while j > -1:
      dof_i.append(i)
      dof_j.append(j)
      j = m.dof_parentid[j]

  for i in range(batch_size):
    d_i = torch.utils._pytree.tree_map(lambda x, i=i: x[i], d) if batched else d
    result_i = result[i] if batched else result
    ncon = (d_i.contact.dist <= 0).sum()
    efc_active = (d_i.efc_J != 0).any(dim=1)
    efc_con = efc_type == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    nefc, nc = int(efc_active.sum()), int((efc_active & efc_con).sum())
    result_i.nnzJ = nefc * m.nv
    if ncon != result_i.ncon or nefc != result_i.nefc:
      mujoco._functions._realloc_con_efc(result_i, ncon=ncon, nefc=nefc)  # pylint: disable=protected-access
    result_i.efc_J_rownnz[:] = np.repeat(m.nv, nefc)
    result_i.efc_J_rowadr[:] = np.arange(0, nefc * m.nv, m.nv)
    result_i.efc_J_colind[:] = np.tile(np.arange(m.nv), nefc)

    for field in types.Data.fields():
      if field.name == 'contact':
        _get_contact(result_i.contact, d_i.contact, nefc - nc)
        continue

      value = getattr(d_i, field.name)

      if field.name in ('xmat', 'ximat', 'geom_xmat', 'site_xmat', 'cam_xmat'):
        value = value.reshape((-1, 9))

      if field.name in ('efc_frictionloss', 'efc_D', 'efc_aref', 'efc_force'):
        value = value[efc_active]

      if field.name == 'efc_J':
        value = value[efc_active].reshape(-1)

      if field.name == 'qM' and not support.is_sparse(m):
        value = value[dof_i, dof_j]

      if field.name == 'qLD' and not support.is_sparse(m):
        value = value[dof_i, dof_j]

      if field.name == 'qLDiagInv' and not support.is_sparse(m):
        value = np.ones(m.nv)

      if value.shape:
        getattr(result_i, field.name)[:] = value
      else:
        setattr(result_i, field.name, value)

    result_i.efc_type[:] = efc_type[efc_active]


def _put_contact(
    c: mujoco._structs._MjContactList, ncon: int, device=None
) -> types.Contact:
  """Puts mujoco.structs._MjContactList onto a device, resulting in mujoco_torch.Contact."""
  fields = {
      f.name: copy.copy(getattr(c, f.name)) for f in types.Contact.fields()
  }
  fields['frame'] = fields['frame'].reshape((-1, 3, 3))
  pad_size = ncon - c.dist.shape[0]
  pad_fn = lambda x: np.concatenate(
      (x, np.zeros((pad_size,) + x.shape[1:], dtype=x.dtype))
  )
  fields = torch.utils._pytree.tree_map(pad_fn, fields)
  fields['dist'][-pad_size:] = np.inf
  fields = TensorDict(fields).to(device=device)

  return types.Contact(**fields)


def put_data(m: mujoco.MjModel, d: mujoco.MjData, device=None) -> types.Data:
  """Puts mujoco.MjData onto a device, resulting in mujoco_torch.Data."""
  ncon = collision_driver.ncon(m)
  ne, nf, nl, nc = constraint.count_constraints(m)
  nefc = ne + nf + nl + nc

  for d_val, val, name in (
      (d.ncon, ncon, 'ncon'),
      (d.ne, ne, 'ne'),
      (d.nf, nf, 'nf'),
      (d.nl, nl, 'nl'),
      (d.nefc, nefc, 'nefc'),
  ):
    if d_val > val:
      raise ValueError(f'd.{name} too high, d.{name} = {d_val}, model = {val}')

  fields = {
      f.name: copy.copy(getattr(d, f.name))  # copy because device_put is async
      for f in types.Data.fields()
      if f.type is torch.Tensor
  }

  for fname in ('xmat', 'ximat', 'geom_xmat', 'site_xmat', 'cam_xmat'):
    fields[fname] = fields[fname].reshape((-1, 3, 3))

  # MJX does not support islanding, so only transfer the first solver_niter
  fields['solver_niter'] = fields['solver_niter'][0]

  # pad efc fields: MuJoCo efc arrays are sparse for inactive constraints.
  # efc_J is also optionally column-sparse (typically for large nv).  MJX is
  # neither: it contains zeros for inactive constraints, and efc_J is always
  # (nefc, nv).  this may change in the future.
  if mujoco.mj_isSparse(m):
    nr = d.efc_J_rownnz.shape[0]
    efc_j = np.zeros((nr, m.nv))
    for i in range(nr):
      rowadr = d.efc_J_rowadr[i]
      for j in range(d.efc_J_rownnz[i]):
        efc_j[i, d.efc_J_colind[rowadr + j]] = fields['efc_J'][rowadr + j]
    fields['efc_J'] = efc_j
  else:
    fields['efc_J'] = fields['efc_J'].reshape((-1 if m.nv else 0, m.nv))

  for fname in ('efc_J', 'efc_frictionloss', 'efc_D', 'efc_aref', 'efc_force'):
    value = np.zeros((nefc, m.nv)) if fname == 'efc_J' else np.zeros(nefc)
    for i in range(4):
      value_beg = sum([ne, nf, nl][:i])
      d_beg = sum([d.ne, d.nf, d.nl][:i])
      size = [d.ne, d.nf, d.nl, d.nefc - d.nl - d.nf - d.ne][i]
      value[value_beg:value_beg+size] = fields[fname][d_beg:d_beg+size]
    fields[fname] = value

  # convert qM and qLD if jacobian is dense
  if not support.is_sparse(m):
    fields['qM'] = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, fields['qM'], d.qM)
    # TODO(erikfrey): derive L*L' from L'*D*L instead of recomputing
    try:
      fields['qLD'] = torch.linalg.cholesky(torch.as_tensor(fields['qM']), upper=True)
    except RuntimeError:
      # this happens when qM is empty or unstable simulation
      fields['qLD'] = np.zeros((m.nv, m.nv))
    fields['qLDiagInv'] = np.zeros(0)

  fields = TensorDict(fields, device=device)
  fields['contact'] = _put_contact(d.contact, ncon, device=device)
  return types.Data.from_tensordict(fields)
