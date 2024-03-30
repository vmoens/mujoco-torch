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
"""Core smooth dynamics functions."""

# from torch import numpy as torch
import mujoco
import torch
from mujoco_torch._src import math
from mujoco_torch._src import scan
# pylint: disable=g-importing-member
from mujoco_torch._src.types import Data
from mujoco_torch._src.types import DisableBit
from mujoco_torch._src.types import JointType
from mujoco_torch._src.types import Model


# pylint: enable=g-importing-member


def kinematics(m: Model, d: Data) -> Data:
  """Converts position/velocity from generalized coordinates to maximal."""

  def fn(carry, jnt_typs, jnt_pos, jnt_axis, qpos, qpos0, pos, quat):
    # calculate joint anchors, axes, body pos and quat in global frame
    # also normalize qpos while we're at it

    if carry is not None:
      _, _, _, parent_pos, parent_quat, _ = carry
      pos = parent_pos + math.rotate(pos, parent_quat)
      quat = math.quat_mul(parent_quat, quat)

    anchors, axes = [], []

    qpos_i = 0
    for i, jnt_typ in enumerate(jnt_typs):
      if jnt_typ == JointType.FREE:
        anchor, axis = qpos[qpos_i : qpos_i + 3], torch.tensor([0.0, 0.0, 1.0])
      else:
        anchor = math.rotate(jnt_pos[i], quat) + pos
        axis = math.rotate(jnt_axis[i], quat)
      anchors, axes = anchors + [anchor], axes + [axis]

      if jnt_typ == JointType.FREE:
        pos = qpos[qpos_i : qpos_i + 3]
        quat = math.normalize(qpos[qpos_i + 3 : qpos_i + 7])
        qpos = torch.scatter(input=qpos, dim=0, index=torch.arange(qpos_i + 3, qpos_i + 7), src=quat)
        qpos_i += 7
      elif jnt_typ == JointType.BALL:
        qloc = math.normalize(qpos[qpos_i : qpos_i + 4])
        qpos = torch.scatter(input=qpos, dim=0, index=torch.arange(qpos_i, qpos_i + 4), src=qloc)
        quat = math.quat_mul(quat, qloc)
        pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
        qpos_i += 4
      elif jnt_typ == JointType.HINGE:
        angle = qpos[qpos_i] - qpos0[qpos_i]
        qloc = math.axis_angle_to_quat(jnt_axis[i], angle)
        quat = math.quat_mul(quat, qloc)
        pos = anchor - math.rotate(jnt_pos[i], quat)  # off-center rotation
        qpos_i += 1
      elif jnt_typ == JointType.SLIDE:
        pos += axis * (qpos[qpos_i] - qpos0[qpos_i])
        qpos_i += 1
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

    anchor = torch.stack(anchors) if anchors else torch.empty((0, 3))
    axis = torch.stack(axes) if axes else torch.empty((0, 3))
    mat = math.quat_to_mat(quat)

    return qpos, anchor, axis, pos, quat, mat

  qpos, xanchor, xaxis, xpos, xquat, xmat = scan.body_tree(
      m,
      fn,
      'jjjqqbb',
      'qjjbbb',
      m.jnt_type,
      m.jnt_pos,
      m.jnt_axis,
      d.qpos,
      m.qpos0,
      m.body_pos,
      m.body_quat,
  )

  @torch.vmap
  def local_to_global(pos1, quat1, pos2, quat2):
    pos = pos1 + math.rotate(pos2, quat1)
    mat = math.quat_to_mat(math.quat_mul(quat1, quat2))
    return pos, mat

  # TODO(erikfrey): confirm that quats are more performant for mjx than mats
  xipos, ximat = local_to_global(xpos, xquat, m.body_ipos, m.body_iquat)
  geom_xpos, geom_xmat = local_to_global(
      xpos[m.geom_bodyid], xquat[m.geom_bodyid], m.geom_pos, m.geom_quat
  )

  d = d.replace(qpos=qpos, xanchor=xanchor, xaxis=xaxis, xpos=xpos)
  d = d.replace(xquat=xquat, xmat=xmat, xipos=xipos, ximat=ximat)
  d = d.replace(geom_xpos=geom_xpos, geom_xmat=geom_xmat)

  return d


def com_pos(m: Model, d: Data) -> Data:
  """Maps inertias and motion dofs to global frame centered at subtree-CoM."""

  # calculate center of mass of each subtree
  def subtree_sum(carry, xipos, body_mass):
    pos, mass = xipos * body_mass, body_mass
    if carry is not None:
      subtree_pos, subtree_mass = carry
      pos, mass = pos + subtree_pos, mass + subtree_mass
    return pos, mass

  pos, mass = scan.body_tree(
      m, subtree_sum, 'bb', 'bb', d.xipos, m.body_mass, reverse=True
  )
  cond = torch.tile(mass < torch.tensor(mujoco.mjMINVAL), (3, 1)).T
  subtree_com = torch.where(cond, d.xipos, torch.vmap(torch.divide)(pos, mass))
  d = d.replace(subtree_com=subtree_com)

  # map inertias to frame centered at subtree_com
  @torch.vmap
  def inert_com(inert, ximat, off, mass):
    h = torch.linalg.cross(off.expand(3, 3), -torch.eye(3, dtype=off.dtype))
    inert = ximat @ torch.diag(inert) @ ximat.T + h @ h.T * mass
    # cinert is triu(inert), mass * off, mass
    inert = inert[(torch.tensor([0, 1, 2, 0, 0, 1]), torch.tensor([0, 1, 2, 1, 2, 2]))]
    # return torch.cat([inert, off * mass, torch.expand_dims(mass, 0)])
    return torch.cat([inert, off * mass, torch.unsqueeze(mass, 0)])

  root_com = subtree_com[torch.tensor(m.body_rootid)]
  offset = d.xipos - root_com
  cinert = inert_com(m.body_inertia, d.ximat, offset, m.body_mass)
  d = d.replace(cinert=cinert)

  # map motion dofs to global frame centered at subtree_com
  def cdof_fn(jnt_typs, root_com, xmat, xanchor, xaxis):
    cdofs = []

    dof_com_fn = lambda a, o: torch.cat([a, torch.linalg.cross(a, o)])

    for i, jnt_typ in enumerate(jnt_typs):
      offset = root_com - xanchor[i]
      if jnt_typ == JointType.FREE:
        cdofs.append(torch.eye(6, 6)[3:])  # free translation
        # cdofs.append(torch.eye(3, 6, 3))  # free translation
        cdofs.append(torch.vmap(dof_com_fn, (0, None))(xmat.T, offset))
      elif jnt_typ == JointType.BALL:
        cdofs.append(torch.vmap(dof_com_fn, (0, None))(xmat.T, offset))
      elif jnt_typ == JointType.HINGE:
        cdof = dof_com_fn(xaxis[i], offset)
        cdofs.append(torch.unsqueeze(cdof, 0))
      elif jnt_typ == JointType.SLIDE:
        cdof = torch.cat([torch.zeros((3,)), xaxis[i]])
        cdofs.append(torch.unsqueeze(cdof, 0))
      else:
        raise RuntimeError(f'unrecognized joint type: {jnt_typ}')

    cdof = torch.cat(cdofs) if cdofs else torch.empty((0, 6))

    return cdof

  cdof = scan.flat(
      m,
      cdof_fn,
      'jbbjj',
      'v',
      m.jnt_type,
      root_com,
      d.xmat,
      d.xanchor,
      d.xaxis,
  )
  d = d.replace(cdof=cdof)

  return d


def crb(m: Model, d: Data) -> Data:
  """Runs composite rigid body inertia algorithm."""

  def crb_fn(crb_child, crb_body):
    if crb_child is not None:
      crb_body += crb_child
    return crb_body

  crb_body = scan.body_tree(m, crb_fn, 'b', 'b', d.cinert, reverse=True)
  # crb_body = crb_body.at[0].set(0.0)
  crb_body[0] = 0.0 # or scatter
  d = d.replace(crb=crb_body)

  # TODO(erikfrey): do centralized take fn?
  # crb_dof = torch.take(crb_body, torch.tensor(m.dof_bodyid), axis=0)
  crb_dof = crb_body[torch.tensor(m.dof_bodyid)]
  crb_cdof = torch.vmap(math.inert_mul)(crb_dof, d.cdof)

  dof_i, dof_j, diag = [], [], []
  for i in range(m.nv):
    diag.append(len(dof_i))
    j = i
    while j > -1:
      dof_i, dof_j = dof_i + [i], dof_j + [j]
      j = m.dof_parentid[j]

  crb_codf_i = crb_cdof[torch.tensor(dof_i)]
  cdof_j = d.cdof[torch.tensor(dof_j)]
  qm = torch.vmap(torch.dot)(crb_codf_i, cdof_j)

  # add armature to diagonal
  qm[torch.tensor(diag)] = qm[torch.tensor(diag)] + m.dof_armature # or scatter_add

  d = d.replace(qM=qm)

  return d


def factor_m(
    m: Model,
    d: Data,
    qM: torch.Tensor,  # pylint:disable=invalid-name
) -> Data:
  """Gets sparse L'*D*L factorizaton of inertia-like matrix M, assumed spd."""

  # build up indices for where we will do backwards updates over qLD
  # TODO(erikfrey): do fewer updates by combining non-overlapping ranges
  dof_madr = torch.tensor(m.dof_Madr)
  updates = {}
  madr_ds = []
  for i in range(m.nv):
    madr_d = madr_ij = m.dof_Madr[i]
    j = i
    while True:
      madr_ds.append(madr_d)
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      madr_j_range = tuple(m.dof_Madr[j : j + 2])
      updates.setdefault(madr_j_range, []).append((madr_d, madr_ij))

  qld = qM

  for (out_beg, out_end), vals in sorted(updates.items(), reverse=True):
    madr_d, madr_ij = torch.tensor(vals).T

    @torch.vmap
    def off_diag_fn(madr_d, madr_ij, qld=qld, width=out_end - out_beg):
      # qld_row = torch.lax.dynamic_slice(qld, (madr_ij,), (width,))
      qld_row = dynamic_slice(qld, (madr_ij,), (width,))
      return -(qld_row[0] / torch.gather(qld, 0, madr_d)) * qld_row

    qld_update = torch.sum(off_diag_fn(madr_d, madr_ij), axis=0)
    qld[out_beg:out_end] += qld_update # or scatter_add
    # TODO(erikfrey): determine if this minimum value guarding is necessary:
    # qld = qld.at[dof_madr].set(torch.maximum(qld[dof_madr], _MJ_MINVAL))

  qld_diag = qld[dof_madr]
  qld = (qld / qld[torch.tensor(madr_ds)])
  qld[dof_madr] = qld_diag

  d = d.replace(qLD=qld, qLDiagInv=1 / qld_diag)

  return d


def solve_m(m: Model, d: Data, x: torch.Tensor) -> torch.Tensor:
  """Computes sparse backsubstitution:  x = inv(L'*D*L)*y ."""

  updates_i, updates_j = {}, {}
  for i in range(m.nv):
    madr_ij, j = m.dof_Madr[i], i
    while True:
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      updates_i.setdefault(i, []).append((madr_ij, j))
      updates_j.setdefault(j, []).append((madr_ij, i))

  # x <- inv(L') * x
  for j, vals in sorted(updates_j.items(), reverse=True):
    madr_ij, i = torch.tensor(vals).T
    x = torch.scatter_add(x, index=torch.tensor(j, dtype=torch.long), dim=0, src=-torch.sum(d.qLD[madr_ij] * x[i]))

  # x <- inv(D) * x
  x = x * d.qLDiagInv

  # x <- inv(L) * x
  for i, vals in sorted(updates_i.items()):
    madr_ij, j = torch.tensor(vals).T
    # x = x.at[i].add(-torch.sum(d.qLD[madr_ij] * x[j]))
    x = torch.scatter_add(x, index=torch.tensor(i, dtype=torch.long), dim=0, src=-torch.sum(d.qLD[madr_ij] * x[j]))

  return x


def dense_m(m: Model, d: Data) -> torch.Tensor:
  """Reconstitute dense mass matrix from qM."""

  is_, js, madr_ijs = [], [], []
  for i in range(m.nv):
    madr_ij, j = m.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  i, j, madr_ij = (torch.tensor(x, dtype=torch.int32) for x in (is_, js, madr_ijs))

  mat = torch.zeros((m.nv, m.nv))
  mat[(i, j)] = d.qM[madr_ij]

  # diagonal, upper triangular, lower triangular
  mat = torch.diag(d.qM[torch.tensor(m.dof_Madr)]) + mat + mat.T

  return mat


def mul_m(m: Model, d: Data, vec: torch.Tensor) -> torch.Tensor:
  """Multiply vector by inertia matrix."""

  diag_mul = d.qM[torch.tensor(m.dof_Madr)] * vec

  is_, js, madr_ijs = [], [], []
  for i in range(m.nv):
    madr_ij, j = m.dof_Madr[i], i

    while True:
      madr_ij, j = madr_ij + 1, m.dof_parentid[j]
      if j == -1:
        break
      is_, js, madr_ijs = is_ + [i], js + [j], madr_ijs + [madr_ij]

  i, j, madr_ij = (torch.tensor(x, dtype=torch.int32) for x in (is_, js, madr_ijs))

  out = diag_mul.at[i].add(d.qM[madr_ij] * vec[j])
  out = out.at[j].add(d.qM[madr_ij] * vec[i])

  return out


def com_vel(m: Model, d: Data) -> Data:
  """Computes cvel, cdof_dot."""

  # forward scan down tree: accumulate link center of mass velocity
  def fn(parent, jnt_typs, cdof, qvel):
    cvel = torch.zeros((6,), dtype=cdof.dtype) if parent is None else parent[0]

    cross_fn = torch.vmap(math.motion_cross, (None, 0))
    cdof_x_qvel = torch.vmap(torch.mul, (1, None), (1,))(cdof, qvel)
    # cdof_x_qvel = cdof * qvel

    dof_beg = 0
    cdof_dots = []
    for jnt_typ in jnt_typs:
      dof_end = dof_beg + JointType(jnt_typ).dof_width()
      if jnt_typ == JointType.FREE:
        cvel = cvel + torch.sum(cdof_x_qvel[:3], axis=0)
        cdof_ang_dot = cross_fn(cvel, cdof[3:])
        cvel = cvel + torch.sum(cdof_x_qvel[3:], axis=0)
        cdof_dots.append(torch.cat((torch.zeros((3, 6)), cdof_ang_dot)))
      else:
        cdof_dots.append(cross_fn(cvel, cdof[dof_beg:dof_end]))
        cvel += torch.sum(cdof_x_qvel[dof_beg:dof_end], axis=0)
      dof_beg = dof_end

    cdof_dot = torch.cat(cdof_dots) if cdof_dots else torch.empty((0, 6))
    return cvel, cdof_dot

  cvel, cdof_dot = scan.body_tree(
      m,
      fn,
      'jvv',
      'bv',
      m.jnt_type,
      d.cdof,
      d.qvel,
  )

  d = d.replace(cvel=cvel, cdof_dot=cdof_dot)

  return d


def rne(m: Model, d: Data) -> Data:
  """Computes inverse dynamics using the recursive Newton-Euler algorithm."""
  # forward scan over tree: accumulate link center of mass acceleration
  def cacc_fn(cacc, cdof_dot, qvel):
    if cacc is None:
      if m.opt.disableflags & DisableBit.GRAVITY:
        cacc = torch.zeros((6,))
      else:
        cacc = torch.cat((torch.zeros((3,)), -m.opt.gravity))

    vm = torch.vmap(torch.multiply, (1, None), (1,))(cdof_dot, qvel)
    vm_sum = torch.sum(vm, 0)
    cacc = cacc + vm_sum

    return cacc

  cacc = scan.body_tree(m, cacc_fn, 'vv', 'b', d.cdof_dot, d.qvel)

  def frc(cinert, cacc, cvel):
    frc = math.inert_mul(cinert, cacc)
    frc += math.motion_cross_force(cvel, math.inert_mul(cinert, cvel))

    return frc

  loc_cfrc = torch.vmap(frc)(d.cinert, cacc, d.cvel)

  # backward scan up tree: accumulate body forces
  def cfrc_fn(cfrc_child, cfrc):
    if cfrc_child is not None:
      cfrc += cfrc_child
    return cfrc

  cfrc = scan.body_tree(m, cfrc_fn, 'b', 'b', loc_cfrc, reverse=True)
  qfrc_bias = torch.vmap(torch.dot)(d.cdof, cfrc[torch.tensor(m.dof_bodyid)])

  d = d.replace(qfrc_bias=qfrc_bias)

  return d


def transmission(m: Model, d: Data) -> Data:
  """Computes actuator/transmission lengths and moments."""
  if not m.nu:
    return d

  def fn(gear, jnt_typ, m_i, m_j, qpos):
    # handles joint transmissions only
    if jnt_typ == JointType.FREE:
      length = torch.zeros(1)
      moment = gear
      m_i = torch.repeat(m_i, 6)
      m_j = m_j + torch.arange(6)
    elif jnt_typ == JointType.BALL:
      axis, _ = math.quat_to_axis_angle(qpos)
      length = torch.dot(axis, gear[:3])[None]
      moment = gear[:3]
      m_i = torch.repeat(m_i, 3)
      m_j = m_j + torch.arange(3)
    elif jnt_typ in (JointType.SLIDE, JointType.HINGE):
      length = qpos * gear[0]
      moment = gear[:1]
      m_i, m_j = m_i[None], m_j[None]
    else:
      raise RuntimeError(f'unrecognized joint type: {jnt_typ}')
    return length, moment, m_i, m_j

  length, m_val, m_i, m_j = scan.flat(
      m,
      fn,
      'ujujq',
      'uvvv',
      m.actuator_gear,
      m.jnt_type,
      torch.arange(m.nu),
      torch.array(m.jnt_dofadr),
      d.qpos,
      group_by='u',
  )
  moment = torch.zeros((m.nu, m.nv)).at[m_i, m_j].set(m_val)
  length = length.reshape((m.nu,))
  d = d.replace(actuator_length=length, actuator_moment=moment)
  return d

def dynamic_slice(operand, start_indices, slice_sizes):
  """Torch version of torch.lax.dynamic_slice."""
  # we must assume that even if indices are batched tensors, they only contain one element
  # It isn't necessarily true for start (one can start at index 0 and 1) but it
  # must be true for the slice index (we cannot have a length of 10 in one dim
  # and 11 in the other, the resulting unbatched tensor would not be rectangular).
  from torch._C._functorch import is_batchedtensor

  def make_slice(start, interval):
    if not is_batchedtensor(start):
      return slice(start.item(), (start + interval).item())
    else:
      idx = [start]
      for i in range(1, interval):
        idx.append(start + i)
      idx = torch.stack(idx, -1)
      return idx
  slices = tuple(make_slice(start, size) for start, size in zip(start_indices, slice_sizes))
  return operand[slices]
