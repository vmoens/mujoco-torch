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
"""Base types used in mujoco_torch."""

import enum
from typing import List, Optional

import torch
import mujoco
import numpy as np
import tensordict


class AutoConvertIntEnum(enum.IntEnum):

  @classmethod
  def _missing_(cls, value):
    if isinstance(value, torch.Tensor):
      return cls(value.item())
    return super()._missing_(value)

class AutoConvertIntFlag(enum.IntFlag):

  @classmethod
  def _missing_(cls, value):
    if isinstance(value, torch.Tensor):
      return cls(value.item())
    return super()._missing_(value)


class NonVmappableTensor(torch.Tensor):
  # this is needed such that tensor[NonVmappableTensor(...)] returns a tensor
  @classmethod
  def __torch_function__(cls, func, types, args=(), kwargs=None):
    if func is torch.Tensor.__getitem__ and not isinstance(args[0], NonVmappableTensor):
      return torch._C._disabled_torch_function_impl(func, types, args)
    if kwargs is None:
      kwargs = {}
    return super().__torch_function__(func, types, args, kwargs)


def is_leaf_vmap(tensortype):
  if not isinstance(tensortype, type):
    return is_leaf_vmap(type(tensortype))
  if issubclass(tensortype, torch.Tensor) and not issubclass(tensortype, NonVmappableTensor):
    return True
  return False

class DisableBit(AutoConvertIntFlag):
  """Disable default feature bitflags.

  Attributes:
    CONSTRAINT:   entire constraint solver
    EQUALITY:     equality constraints
    FRICTIONLOSS: joint and tendon frictionloss constraints
    LIMIT:        joint and tendon limit constraints
    CONTACT:      contact constraints
    PASSIVE:      passive forces
    GRAVITY:      gravitational forces
    CLAMPCTRL:    clamp control to specified range
    WARMSTART:    warmstart constraint solver
    ACTUATION:    apply actuation forces
    REFSAFE:      integrator safety: make ref[0]>=2*timestep
  """
  CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
  EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
  LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
  CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
  PASSIVE = mujoco.mjtDisableBit.mjDSBL_PASSIVE
  GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
  CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
  WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
  ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
  REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
  EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
  FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
  # unsupported: FRICTIONLOSS, SENSOR, MIDPHASE


class JointType(AutoConvertIntEnum):
  """Type of degree of freedom.

  Attributes:
    FREE:  global position and orientation (quat)       (7,)
    BALL:  orientation (quat) relative to parent        (4,)
    SLIDE: sliding distance along body-fixed axis       (1,)
    HINGE: rotation angle (rad) around body-fixed axis  (1,)
  """
  FREE = mujoco.mjtJoint.mjJNT_FREE
  BALL = mujoco.mjtJoint.mjJNT_BALL
  SLIDE = mujoco.mjtJoint.mjJNT_SLIDE
  HINGE = mujoco.mjtJoint.mjJNT_HINGE

  def dof_width(self) -> int:
    return {0: 6, 1: 3, 2: 1, 3: 1}[self.value]

  def qpos_width(self) -> int:
    return {0: 7, 1: 4, 2: 1, 3: 1}[self.value]


class IntegratorType(AutoConvertIntEnum):
  """Integrator mode.

  Attributes:
    EULER: semi-implicit Euler
    RK4: 4th-order Runge Kutta
  """

  EULER = mujoco.mjtIntegrator.mjINT_EULER
  RK4 = mujoco.mjtIntegrator.mjINT_RK4
  # unsupported: IMPLICIT, IMPLICITFAST

  @classmethod
  def _missing_(cls, value):
    return cls(value.item())



class GeomType(AutoConvertIntEnum):
  """Type of geometry.

  Attributes:
    PLANE: plane
    HFIELD: height field
    SPHERE: sphere
    CAPSULE: capsule
    ELLIPSOID: ellipsoid
    CYLINDER: cylinder
    BOX: box
    MESH: mesh
  """

  PLANE = mujoco.mjtGeom.mjGEOM_PLANE
  HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
  SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
  CAPSULE = mujoco.mjtGeom.mjGEOM_CAPSULE
  ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  CYLINDER = mujoco.mjtGeom.mjGEOM_CYLINDER
  BOX = mujoco.mjtGeom.mjGEOM_BOX
  MESH = mujoco.mjtGeom.mjGEOM_MESH
  # unsupported: NGEOMTYPES, ARROW*, LINE, SKIN, LABEL, NONE


class ConeType(AutoConvertIntEnum):
  """Type of friction cone.

  Attributes:
    PYRAMIDAL: pyramidal
  """
  PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL
  # unsupported: ELLIPTIC


class JacobianType(AutoConvertIntEnum):
  """Type of constraint Jacobian.

  Attributes:
    DENSE: dense
    SPARSE: sparse
    AUTO: sparse if nv>60 and device is TPU, dense otherwise
  """
  DENSE = mujoco.mjtJacobian.mjJAC_DENSE
  SPARSE = mujoco.mjtJacobian.mjJAC_SPARSE
  AUTO = mujoco.mjtJacobian.mjJAC_AUTO


class SolverType(AutoConvertIntEnum):
  """Constraint solver algorithm.

  Attributes:
    CG: Conjugate gradient (primal)
  """
  # unsupported: PGS
  CG = mujoco.mjtSolver.mjSOL_CG
  NEWTON = mujoco.mjtSolver.mjSOL_NEWTON


class EqType(AutoConvertIntEnum):
  """Type of equality constraint.

  Attributes:
    CONNECT: connect two bodies at a point (ball joint)
    WELD: fix relative position and orientation of two bodies
    JOINT: couple the values of two scalar joints with cubic
  """
  CONNECT = mujoco.mjtEq.mjEQ_CONNECT
  WELD = mujoco.mjtEq.mjEQ_WELD
  JOINT = mujoco.mjtEq.mjEQ_JOINT
  # unsupported: TENDON, DISTANCE


class TrnType(AutoConvertIntEnum):
  """Type of actuator transmission.

  Attributes:
    JOINT: force on joint
    SITE: force on site
  """
  JOINT = mujoco.mjtTrn.mjTRN_JOINT
  SITE = mujoco.mjtTrn.mjTRN_SITE
  # unsupported: JOINTINPARENT, SLIDERCRANK, TENDON, BODY


class DynType(AutoConvertIntEnum):
  """Type of actuator dynamics.

  Attributes:
    NONE: no internal dynamics; ctrl specifies force
    INTEGRATOR: integrator: da/dt = u
    FILTER: linear filter: da/dt = (u-a) / tau
    FILTEREXACT: linear filter: da/dt = (u-a) / tau, with exact integration
  """
  NONE = mujoco.mjtDyn.mjDYN_NONE
  INTEGRATOR = mujoco.mjtDyn.mjDYN_INTEGRATOR
  FILTER = mujoco.mjtDyn.mjDYN_FILTER
  FILTEREXACT = mujoco.mjtDyn.mjDYN_FILTEREXACT
  # unsupported: MUSCLE, USER


class GainType(AutoConvertIntEnum):
  """Type of actuator gain.

  Attributes:
    FIXED: fixed gain
    AFFINE: const + kp*length + kv*velocity
  """
  FIXED = mujoco.mjtGain.mjGAIN_FIXED
  AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
  # unsupported: MUSCLE, USER


class BiasType(AutoConvertIntEnum):
  """Type of actuator bias.

  Attributes:
    NONE: no bias
    AFFINE: const + kp*length + kv*velocity
  """
  NONE = mujoco.mjtBias.mjBIAS_NONE
  AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
  # unsupported: MUSCLE, USER


class CamLightType(AutoConvertIntEnum):
  """Type of camera light.

  Attributes:
    FIXED: pos and rot fixed in body
    TRACK: pos tracks body, rot fixed in global
    TRACKCOM: pos tracks subtree com, rot fixed in body
    TARGETBODY: pos fixed in body, rot tracks target body
    TARGETBODYCOM: pos fixed in body, rot tracks target subtree com
  """

  FIXED = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
  TRACK = mujoco.mjtCamLight.mjCAMLIGHT_TRACK
  TRACKCOM = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
  TARGETBODY = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
  TARGETBODYCOM = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM

@tensordict.tensorclass(autocast=True)
class Option:
  """Physics options.

  Attributes:
    timestep:         timestep
    impratio:         ratio of friction-to-normal contact impedance
    tolerance:        main solver tolerance
    ls_tolerance:     CG/Newton linesearch tolerance
    gravity:          gravitational acceleration                 (3,)
    wind:             wind (for lift, drag and viscosity)
    density:          density of medium
    viscosity:        viscosity of medium
    has_fluid_params: automatically set by mjx if wind/density/viscosity are
      nonzero. Not used by mj
    integrator:       integration mode
    cone:             type of friction cone
    jacobian:         matrix layout for mass matrices (dense or sparse)
                      (note that this is different from MuJoCo, where jacobian
                      specifies whether efc_J and its accompanying matrices
                      are dense or sparse.
    solver:           solver algorithm
    iterations:       number of main solver iterations
    ls_iterations:    maximum number of CG/Newton linesearch iterations
    disableflags:     bit flags for disabling standard features
  """
  timestep: torch.Tensor
  # unsupported: apirate
  impratio: torch.Tensor
  tolerance: torch.Tensor
  ls_tolerance: torch.Tensor
  # unsupported: noslip_tolerance, mpr_tolerance
  gravity: torch.Tensor
  wind: torch.Tensor
  density: torch.Tensor
  viscosity: torch.Tensor
  has_fluid_params: bool
  # unsupported: magnetic, o_margin, o_solref, o_solimp
  integrator: IntegratorType
  cone: ConeType
  jacobian: JacobianType
  solver: SolverType
  iterations: int
  ls_iterations: int
  # unsupported: noslip_iterations, mpr_iterations
  disableflags: DisableBit
  # unsupported: enableflags


@tensordict.tensorclass(autocast=True)
class Statistic:
  """Model statistics (in qpos0).

  Attributes:
    meaninertia: mean diagonal inertia
  """
  meaninertia: torch.Tensor
  # unsupported: meanmass, meansize, extent, center


@tensordict.tensorclass(autocast=True)
class Model:
  """Static model of the scene that remains unchanged with each physics step.

  Attributes:
    nq: number of generalized coordinates = dim(qpos)
    nv: number of degrees of freedom = dim(qvel)
    nu: number of actuators/controls = dim(ctrl)
    na: number of activation states = dim(act)
    nbody: number of bodies
    njnt: number of joints
    ngeom: number of geoms
    nsite: number of sites
    ncam: number of cameras
    nmesh: number of meshes
    nmeshvert: number of vertices in all meshes
    nmeshface: number of triangular faces in all meshes
    nmat: number of materials
    npair: number of predefined geom pairs
    nexclude: number of excluded geom pairs
    neq: number of equality constraints
    nnumeric: number of numeric custom fields
    nuserdata: size of userdata array
    nM: number of non-zeros in sparse inertia matrix
    opt: physics options
    stat: model statistics
    qpos0: qpos values at default pose                        (nq,)
    qpos_spring: reference pose for springs                   (nq,)
    body_parentid: id of body's parent                        (nbody,)
    body_rootid: id of root above body                        (nbody,)
    body_weldid: id of body that this body is welded to       (nbody,)
    body_jntnum: number of joints for this body               (nbody,)
    body_jntadr: start addr of joints; -1: no joints          (nbody,)
    body_dofnum: number of motion degrees of freedom          (nbody,)
    body_dofadr: start addr of dofs; -1: no dofs              (nbody,)
    body_geomnum: number of geoms                             (nbody,)
    body_geomadr: start addr of geoms; -1: no geoms           (nbody,)
    body_pos: position offset rel. to parent body             (nbody, 3)
    body_quat: orientation offset rel. to parent body         (nbody, 4)
    body_ipos: local position of center of mass               (nbody, 3)
    body_iquat: local orientation of inertia ellipsoid        (nbody, 4)
    body_mass: mass                                           (nbody,)
    body_subtreemass: mass of subtree starting at this body   (nbody,)
    body_inertia: diagonal inertia in ipos/iquat frame        (nbody, 3)
    body_invweight0: mean inv inert in qpos0 (trn, rot)       (nbody, 2)
    jnt_type: type of joint (mjtJoint)                        (njnt,)
    jnt_qposadr: start addr in 'qpos' for joint's data        (njnt,)
    jnt_dofadr: start addr in 'qvel' for joint's data         (njnt,)
    jnt_bodyid: id of joint's body                            (njnt,)
    jnt_group: group for visibility                           (njnt,)
    jnt_limited: does joint have limits                       (njnt,)
    jnt_solref: constraint solver reference: limit            (njnt, mjNREF)
    jnt_solimp: constraint solver impedance: limit            (njnt, mjNIMP)
    jnt_pos: local anchor position                            (njnt, 3)
    jnt_axis: local joint axis                                (njnt, 3)
    jnt_stiffness: stiffness coefficient                      (njnt,)
    jnt_range: joint limits                                   (njnt, 2)
    jnt_actfrcrange: range of total actuator force            (njnt, 2)
    jnt_margin: min distance for limit detection              (njnt,)
    dof_bodyid: id of dof's body                              (nv,)
    dof_jntid: id of dof's joint                              (nv,)
    dof_parentid: id of dof's parent; -1: none                (nv,)
    dof_Madr: dof address in M-diagonal                       (nv,)
    dof_solref: constraint solver reference:frictionloss      (nv, mjNREF)
    dof_solimp: constraint solver impedance:frictionloss      (nv, mjNIMP)
    dof_frictionloss: dof friction loss                       (nv,)
    dof_armature: dof armature inertia/mass                   (nv,)
    dof_damping: damping coefficient                          (nv,)
    dof_invweight0: diag. inverse inertia in qpos0            (nv,)
    dof_M0: diag. inertia in qpos0                            (nv,)
    geom_type: geometric type (mjtGeom)                       (ngeom,)
    geom_contype: geom contact type                           (ngeom,)
    geom_conaffinity: geom contact affinity                   (ngeom,)
    geom_condim: contact dimensionality (1, 3, 4, 6)          (ngeom,)
    geom_bodyid: id of geom's body                            (ngeom,)
    geom_dataid: id of geom's mesh/hfield; -1: none           (ngeom,)
    geom_group: group for visibility                          (ngeom,)
    geom_matid: material id for rendering                     (ngeom,)
    geom_priority: geom contact priority                      (ngeom,)
    geom_solmix: mixing coef for solref/imp in geom pair      (ngeom,)
    geom_solref: constraint solver reference: contact         (ngeom, mjNREF)
    geom_solimp: constraint solver impedance: contact         (ngeom, mjNIMP)
    geom_size: geom-specific size parameters                  (ngeom, 3)
    geom_pos: local position offset rel. to body              (ngeom, 3)
    geom_quat: local orientation offset rel. to body          (ngeom, 4)
    geom_friction: friction for (slide, spin, roll)           (ngeom, 3)
    geom_margin: include in solver if dist<margin-gap         (ngeom,)
    geom_gap: include in solver if dist<margin-gap            (ngeom,)
    geom_rgba: rgba when material is omitted                  (ngeom, 4)
    site_bodyid: id of site's body                            (nsite,)
    site_pos: local position offset rel. to body              (nsite, 3)
    site_quat: local orientation offset rel. to body          (nsite, 4)
    cam_mode:  camera tracking mode (mjtCamLight)             (ncam,)
    cam_bodyid:  id of camera's body                          (ncam,)
    cam_targetbodyid:  id of targeted body; -1: none          (ncam,)
    cam_pos:  position rel. to body frame                     (ncam, 3)
    cam_quat:  orientation rel. to body frame                 (ncam, 4)
    cam_poscom0:  global position rel. to sub-com in qpos0    (ncam, 3)
    cam_pos0:  global position rel. to body in qpos0          (ncam, 3)
    cam_mat0:  global orientation in qpos0                    (ncam, 9)
    mat_rgba: rgba                                            (nmat, 4)
    mesh_vertadr: first vertex address                        (nmesh x 1)
    mesh_faceadr: first face address                          (nmesh x 1)
    mesh_vert: vertex positions for all meshes                (nmeshvert, 3)
    mesh_face: vertex face data                               (nmeshface, 3)
    geom_convex_face: vertex face data, MJX only              (ngeom,)
    geom_convex_vert: vertex data, MJX only                   (ngeom,)
    geom_convex_edge_dir: unique edge direction, MJX only     (ngeom,)
    geom_convex_facenormal: normal face data, MJX only        (ngeom,)
    geom_convex_face_edge: edges for each face                (ngeom,)
    geom_convex_face_edge_normal: edge normals for each face  (ngeom,)
    pair_dim: contact dimensionality                          (npair,)
    pair_geom1: id of geom1                                   (npair,)
    pair_geom2: id of geom2                                   (npair,)
    pair_solref: solver reference: contact normal             (npair, mjNREF)
    pair_solreffriction: solver reference: contact friction   (npair, mjNREF)
    pair_solimp: solver impedance: contact                    (npair, mjNIMP)
    pair_margin: include in solver if dist<margin-gap         (npair,)
    pair_gap: include in solver if dist<margin-gap            (npair,)
    pair_friction: tangent1, 2, spin, roll1, 2                (npair, 5)
    exclude_signature: (body1+1) << 16 + body2+1              (nexclude,)
    eq_type: constraint type (mjtEq)                          (neq,)
    eq_obj1id: id of object 1                                 (neq,)
    eq_obj2id: id of object 2                                 (neq,)
    eq_active0: initial enable/disable constraint state       (neq,)
    eq_solref: constraint solver reference                    (neq, mjNREF)
    eq_solimp: constraint solver impedance                    (neq, mjNIMP)
    eq_data: numeric data for constraint                      (neq, mjNEQDATA)
    actuator_trntype: transmission type (mjtTrn)              (nu,)
    actuator_dyntype: dynamics type (mjtDyn)                  (nu,)
    actuator_gaintype: gain type (mjtGain)                    (nu,)
    actuator_biastype: bias type (mjtBias)                    (nu,)
    actuator_trnid: transmission id: joint, tendon, site      (nu, 2)
    actuator_actadr: first activation address; -1: stateless  (nu,)
    actuator_actnum: number of activation variables           (nu,)
    actuator_ctrllimited: is control limited                  (nu,)
    actuator_forcelimited: is force limited                   (nu,)
    actuator_actlimited: is activation limited                (nu,)
    actuator_dynprm: dynamics parameters                      (nu, mjNDYN)
    actuator_gainprm: gain parameters                         (nu, mjNGAIN)
    actuator_biasprm: bias parameters                         (nu, mjNBIAS)
    actuator_ctrlrange: range of controls                     (nu, 2)
    actuator_forcerange: range of forces                      (nu, 2)
    actuator_actrange: range of activations                   (nu, 2)
    actuator_gear: scale length and transmitted force         (nu, 6)
    numeric_adr: address of field in numeric_data             (nnumeric,)
    numeric_data: array of all numeric fields                 (nnumericdata,)
    name_numericadr: numeric name pointers                    (nnumeric,)
    obj_names: names of all objects, 0-terminated                 (nnames,)
  """
  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  ncam: int
  nmesh: int
  nmeshvert: int
  nmeshface: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  nnumeric: int
  nuserdata: int
  nM: int  # pylint:disable=invalid-name
  opt: Option
  stat: Statistic
  qpos0: torch.Tensor
  qpos_spring: torch.Tensor
  body_parentid: torch.Tensor
  body_rootid: torch.Tensor
  body_weldid: torch.Tensor
  body_jntnum: torch.Tensor
  body_jntadr: torch.Tensor
  body_dofnum: torch.Tensor
  body_dofadr: torch.Tensor
  body_geomnum: torch.Tensor
  body_geomadr: torch.Tensor
  body_pos: torch.Tensor
  body_quat: torch.Tensor
  body_ipos: torch.Tensor
  body_iquat: torch.Tensor
  body_mass: torch.Tensor
  body_subtreemass: torch.Tensor
  body_inertia: torch.Tensor
  body_invweight0: torch.Tensor
  jnt_type: NonVmappableTensor
  jnt_qposadr: NonVmappableTensor
  jnt_dofadr: NonVmappableTensor
  jnt_bodyid: NonVmappableTensor
  jnt_limited: NonVmappableTensor
  jnt_actfrclimited: NonVmappableTensor
  jnt_solref: torch.Tensor
  jnt_solimp: torch.Tensor
  jnt_pos: torch.Tensor
  jnt_axis: torch.Tensor
  jnt_stiffness: torch.Tensor
  jnt_range: torch.Tensor
  jnt_actfrcrange: torch.Tensor
  jnt_margin: torch.Tensor
  dof_bodyid: NonVmappableTensor
  dof_jntid: NonVmappableTensor
  dof_parentid: NonVmappableTensor
  dof_Madr: NonVmappableTensor  # pylint:disable=invalid-name
  dof_solref: torch.Tensor
  dof_solimp: torch.Tensor
  dof_frictionloss: torch.Tensor
  dof_armature: torch.Tensor
  dof_damping: torch.Tensor
  dof_invweight0: torch.Tensor
  dof_M0: torch.Tensor  # pylint:disable=invalid-name
  geom_type: NonVmappableTensor
  geom_contype: NonVmappableTensor
  geom_conaffinity: NonVmappableTensor
  geom_condim: NonVmappableTensor
  geom_bodyid: NonVmappableTensor
  geom_dataid: NonVmappableTensor
  geom_group: NonVmappableTensor
  geom_matid: NonVmappableTensor
  geom_priority: NonVmappableTensor
  geom_solmix: torch.Tensor
  geom_solref: torch.Tensor
  geom_solimp: torch.Tensor
  geom_size: torch.Tensor
  geom_pos: torch.Tensor
  geom_quat: torch.Tensor
  geom_friction: torch.Tensor
  geom_margin: torch.Tensor
  geom_gap: torch.Tensor
  geom_rgba: NonVmappableTensor
  site_bodyid: NonVmappableTensor
  site_pos: torch.Tensor
  site_quat: torch.Tensor
  cam_mode: NonVmappableTensor
  cam_bodyid: NonVmappableTensor
  cam_targetbodyid: NonVmappableTensor
  cam_pos: torch.Tensor
  cam_quat: torch.Tensor
  cam_poscom0: torch.Tensor
  cam_pos0: torch.Tensor
  cam_mat0: torch.Tensor
  mesh_vertadr: NonVmappableTensor
  mesh_faceadr: NonVmappableTensor
  mesh_vert: NonVmappableTensor
  mesh_face: NonVmappableTensor
  mat_rgba: NonVmappableTensor
  pair_dim: NonVmappableTensor
  pair_geom1: NonVmappableTensor
  pair_geom2: NonVmappableTensor

  geom_convex_face: NonVmappableTensor  # List[Optional[torch.Tensor]]
  geom_convex_vert: NonVmappableTensor  # List[Optional[torch.Tensor]]
  geom_convex_edge_dir: NonVmappableTensor  # List[Optional[torch.Tensor]]
  geom_convex_facenormal: NonVmappableTensor  # List[Optional[torch.Tensor]]
  geom_convex_edge: NonVmappableTensor  # List[Optional[torch.Tensor]]
  geom_convex_edge_face_normal: NonVmappableTensor  # List[Optional[torch.Tensor]]

  pair_solref: torch.Tensor
  pair_solreffriction: torch.Tensor
  pair_solimp: torch.Tensor
  pair_margin: torch.Tensor
  pair_gap: torch.Tensor
  pair_friction: torch.Tensor
  exclude_signature: NonVmappableTensor
  eq_type: NonVmappableTensor
  eq_obj1id: NonVmappableTensor
  eq_obj2id: NonVmappableTensor
  eq_active0: NonVmappableTensor
  eq_solref: torch.Tensor
  eq_solimp: torch.Tensor
  eq_data: torch.Tensor
  actuator_trntype: NonVmappableTensor
  actuator_dyntype: NonVmappableTensor
  actuator_gaintype: NonVmappableTensor
  actuator_biastype: NonVmappableTensor
  actuator_trnid: NonVmappableTensor
  actuator_actadr: NonVmappableTensor
  actuator_actnum: NonVmappableTensor
  actuator_ctrllimited: NonVmappableTensor
  actuator_forcelimited: NonVmappableTensor
  actuator_actlimited: NonVmappableTensor
  actuator_dynprm: torch.Tensor
  actuator_gainprm: torch.Tensor
  actuator_biasprm: torch.Tensor
  actuator_ctrlrange: torch.Tensor
  actuator_forcerange: torch.Tensor
  actuator_actrange: torch.Tensor
  actuator_gear: torch.Tensor
  numeric_adr: NonVmappableTensor
  numeric_data: NonVmappableTensor
  name_numericadr: NonVmappableTensor
  _names: bytes = None

@tensordict.tensorclass(autocast=True)
class Contact:
  """Result of collision detection functions.

  Attributes:
    dist: distance between nearest points; neg: penetration
    pos: position of contact point: midpoint between geoms            (3,)
    frame: normal is in [0-2]                                         (9,)
    includemargin: include if dist<includemargin=margin-gap           (1,)
    friction: tangent1, 2, spin, roll1, 2                             (5,)
    solref: constraint solver reference, normal direction             (mjNREF,)
    solreffriction: constraint solver reference, friction directions  (mjNREF,)
    solimp: constraint solver impedance                               (mjNIMP,)
    geom1: id of geom 1
    geom2: id of geom 2
  """
  dist: torch.Tensor
  pos: torch.Tensor
  frame: torch.Tensor
  includemargin: torch.Tensor
  friction: torch.Tensor
  solref: torch.Tensor
  solreffriction: torch.Tensor
  solimp: torch.Tensor
  # unsupported: mu, H, dim
  geom1: torch.Tensor
  geom2: torch.Tensor
  # unsupported: efc_address, exclude

  @classmethod
  def zero(cls, ncon: int = 0) -> 'Contact':
    """Returns a contact filled with zeros."""
    return Contact(
        dist=torch.zeros(ncon),
        pos=torch.zeros((ncon, 3,)),
        frame=torch.zeros((ncon, 3, 3)),
        includemargin=torch.zeros(ncon),
        friction=torch.zeros((ncon, 5)),
        solref=torch.zeros((ncon, mujoco.mjNREF)),
        solreffriction=torch.zeros((ncon, mujoco.mjNREF)),
        solimp=torch.zeros((ncon, mujoco.mjNIMP,)),
        geom1=torch.zeros(ncon, dtype=int),
        geom2=torch.zeros(ncon, dtype=int),
    )


@tensordict.tensorclass(autocast=True)
class Data:
  r"""Dynamic state that updates each step.\

  Attributes:
    solver_niter: number of solver iterations, per island         (mjNISLAND,)
    time: simulation time
    qpos: position                                                (nq,)
    qvel: velocity                                                (nv,)
    act: actuator activation                                      (na,)
    qacc_warmstart: acceleration used for warmstart               (nv,)
    ctrl: control                                                 (nu,)
    qfrc_applied: applied generalized force                       (nv,)
    xfrc_applied: applied Cartesian force/torque                  (nbody, 6)
    eq_active: enable/disable constraints                         (neq,)
    qacc: acceleration                                            (nv,)
    act_dot: time-derivative of actuator activation               (na,)
    xpos:  Cartesian position of body frame                       (nbody, 3)
    xquat: Cartesian orientation of body frame                    (nbody, 4)
    xmat:  Cartesian orientation of body frame                    (nbody, 3, 3)
    xipos: Cartesian position of body com                         (nbody, 3)
    ximat: Cartesian orientation of body inertia                  (nbody, 3, 3)
    xanchor: Cartesian position of joint anchor                   (njnt, 3)
    xaxis: Cartesian joint axis                                   (njnt, 3)
    geom_xpos: Cartesian geom position                            (ngeom, 3)
    geom_xmat: Cartesian geom orientation                         (ngeom, 3, 3)
    site_xpos: Cartesian site position                            (nsite, 3)
    site_xmat: Cartesian site orientation                         (nsite, 9)
    cam_xpos: Cartesian camera position                           (ncam, 3)
    cam_xmat: Cartesian camera orientation                        (ncam, 9)
    subtree_com: center of mass of each subtree                   (nbody, 3)
    cdof: com-based motion axis of each dof                       (nv, 6)
    cinert: com-based body inertia and mass                       (nbody, 10)
    actuator_length: actuator lengths                             (nu,)
    actuator_moment: actuator moments                             (nu, nv)
    crb: com-based composite inertia and mass                     (nbody, 10)
    qM: total inertia                                  if sparse: (nM,)
                                                       if dense:  (nv, nv)
    qLD: L'*D*L (or Cholesky) factorization of M.      if sparse: (nM,)
                                                       if dense:  (nv, nv)
    qLDiagInv: 1/diag(D)                               if sparse: (nv,)
                                                       if dense:  (0,)
    contact: list of all detected contacts                        (ncon,)
    efc_J: constraint Jacobian                                    (nefc, nv)
    efc_frictionloss: frictionloss (friction)                     (nefc,)
    efc_D: constraint mass                                        (nefc,)
    actuator_velocity: actuator velocities                        (nu,)
    cvel: com-based velocity [3D rot; 3D tran]                    (nbody, 6)
    cdof_dot: time-derivative of cdof                             (nv, 6)
    qfrc_bias: C(qpos,qvel)                                       (nv,)
    qfrc_passive: passive force                                   (nv,)
    efc_aref: reference pseudo-acceleration                       (nefc,)
    qfrc_actuator: actuator force                                 (nv,)
    qfrc_smooth: net unconstrained force                          (nv,)
    qacc_smooth: unconstrained acceleration                       (nv,)
    qfrc_constraint: constraint force                             (nv,)
    qfrc_inverse: net external force; should equal:               (nv,)
      qfrc_applied + J'*xfrc_applied + qfrc_actuator
    efc_force: constraint force in constraint space               (nefc,)
    userdata: user data, not touched by engine                    (nuserdata,)
  """
  # solver statistics:
  solver_niter: torch.Tensor
  # global properties:
  time: torch.Tensor
  # state:
  qpos: torch.Tensor
  qvel: torch.Tensor
  act: torch.Tensor
  qacc_warmstart: torch.Tensor
  # control:
  ctrl: torch.Tensor
  qfrc_applied: torch.Tensor
  xfrc_applied: torch.Tensor
  eq_active: torch.Tensor
  # dynamics:
  qacc: torch.Tensor
  act_dot: torch.Tensor
  # user data:
  userdata: torch.Tensor
  # position dependent:
  xpos: torch.Tensor
  xquat: torch.Tensor
  xmat: torch.Tensor
  xipos: torch.Tensor
  ximat: torch.Tensor
  xanchor: torch.Tensor
  xaxis: torch.Tensor
  geom_xpos: torch.Tensor
  geom_xmat: torch.Tensor
  site_xpos: torch.Tensor
  site_xmat: torch.Tensor
  cam_xpos: torch.Tensor
  cam_xmat: torch.Tensor
  subtree_com: torch.Tensor
  cdof: torch.Tensor
  cinert: torch.Tensor
  crb: torch.Tensor
  actuator_length: torch.Tensor
  actuator_moment: torch.Tensor
  qM: torch.Tensor  # pylint:disable=invalid-name
  qLD: torch.Tensor  # pylint:disable=invalid-name
  qLDiagInv: torch.Tensor  # pylint:disable=invalid-name
  contact: Contact
  efc_J: torch.Tensor  # pylint:disable=invalid-name
  efc_frictionloss: torch.Tensor
  efc_D: torch.Tensor  # pylint:disable=invalid-name
  # position, velocity dependent:
  actuator_velocity: torch.Tensor
  cvel: torch.Tensor
  cdof_dot: torch.Tensor
  qfrc_bias: torch.Tensor
  qfrc_passive: torch.Tensor
  efc_aref: torch.Tensor
  # position, velcoity, control & acceleration dependent:
  qfrc_actuator: torch.Tensor
  qfrc_smooth: torch.Tensor
  qacc_smooth: torch.Tensor
  qfrc_constraint: torch.Tensor
  qfrc_inverse: torch.Tensor
  efc_force: torch.Tensor

def pytree_repr(tree):
  def repr_node(node):
    if isinstance(node, torch.Tensor):
      return f"{type(node).__name__}(shape={node.shape}, dtype={node.dtype}, device={node.device}, requires_grad={node.requires_grad})\n"
    if isinstance(node, np.ndarray):
      return f"{type(node).__name__}(shape={node.shape}, dtype=node.dtype)\n"
  return repr(torch.utils._pytree.tree_map(repr_node, tree))
