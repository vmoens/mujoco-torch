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
"""Base types used in MJX, ported to PyTorch."""

import enum

import mujoco
import numpy as np
import torch

from mujoco_torch._src.dataclasses import MjTensorClass  # pylint: disable=g-importing-member


class DisableBit(enum.IntFlag):
    """Disable default feature bitflags.

    Attributes:
      CONSTRAINT:   entire constraint solver
      EQUALITY:     equality constraints
      FRICTIONLOSS: joint and tendon frictionloss constraints
      LIMIT:        joint and tendon limit constraints
      CONTACT:      contact constraints
      SPRING:       passive spring forces
      DAMPER:       passive damper forces
      GRAVITY:      gravitational forces
      CLAMPCTRL:    clamp control to specified range
      WARMSTART:    warmstart constraint solver
      ACTUATION:    apply actuation forces
      REFSAFE:      integrator safety: make ref[0]>=2*timestep
      SENSOR:       sensors
      EULERDAMP:    Euler damping
      FILTERPARENT: filter parent
    """

    CONSTRAINT = mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    EQUALITY = mujoco.mjtDisableBit.mjDSBL_EQUALITY
    FRICTIONLOSS = mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
    LIMIT = mujoco.mjtDisableBit.mjDSBL_LIMIT
    CONTACT = mujoco.mjtDisableBit.mjDSBL_CONTACT
    SPRING = mujoco.mjtDisableBit.mjDSBL_SPRING
    DAMPER = mujoco.mjtDisableBit.mjDSBL_DAMPER
    GRAVITY = mujoco.mjtDisableBit.mjDSBL_GRAVITY
    CLAMPCTRL = mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL
    WARMSTART = mujoco.mjtDisableBit.mjDSBL_WARMSTART
    ACTUATION = mujoco.mjtDisableBit.mjDSBL_ACTUATION
    REFSAFE = mujoco.mjtDisableBit.mjDSBL_REFSAFE
    SENSOR = mujoco.mjtDisableBit.mjDSBL_SENSOR
    EULERDAMP = mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    FILTERPARENT = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    # unsupported: MIDPHASE


class EnableBit(enum.IntFlag):
    """Enable optional feature bitflags.

    Attributes:
      INVDISCRETE: discrete-time inverse dynamics
      MULTICCD: multi-point CCD
      SLEEP: enable sleep
    """

    INVDISCRETE = mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    MULTICCD = mujoco.mjtEnableBit.mjENBL_MULTICCD
    SLEEP = mujoco.mjtEnableBit.mjENBL_SLEEP
    # unsupported: OVERRIDE, ENERGY, FWDINV, ISLAND


class JointType(enum.IntEnum):
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


class IntegratorType(enum.IntEnum):
    """Integrator mode.

    Attributes:
      EULER: semi-implicit Euler
      RK4: 4th-order Runge Kutta
      IMPLICITFAST: implicit in velocity, no rne derivative
    """

    EULER = mujoco.mjtIntegrator.mjINT_EULER
    RK4 = mujoco.mjtIntegrator.mjINT_RK4
    IMPLICITFAST = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # unsupported: IMPLICIT


class GeomType(enum.IntEnum):
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


class ConvexMesh(MjTensorClass):
    """Geom properties for convex meshes.

    Attributes:
      vert: vertices of the convex mesh
      face: faces of the convex mesh (materialized vertex positions)
      face_normal: normal vectors for the faces
      edge: edge indexes for all edges in the convex mesh
      edge_face_normal: face normals adjacent to edges in `edge`
    """

    vert: torch.Tensor
    face: torch.Tensor
    face_normal: torch.Tensor
    edge: torch.Tensor
    edge_face_normal: torch.Tensor

    @property
    def facenorm(self) -> torch.Tensor:
        """Alias for face_normal for collision API compatibility."""
        return self.face_normal


class ConeType(enum.IntEnum):
    """Type of friction cone.

    Attributes:
      PYRAMIDAL: pyramidal
      ELLIPTIC: elliptic
    """

    PYRAMIDAL = mujoco.mjtCone.mjCONE_PYRAMIDAL


class JacobianType(enum.IntEnum):
    """Type of constraint Jacobian.

    Attributes:
      DENSE: dense
      SPARSE: sparse
      AUTO: determined automatically
    """

    DENSE = mujoco.mjtJacobian.mjJAC_DENSE
    SPARSE = mujoco.mjtJacobian.mjJAC_SPARSE
    AUTO = mujoco.mjtJacobian.mjJAC_AUTO


class SolverType(enum.IntEnum):
    """Constraint solver algorithm.

    Attributes:
      CG: Conjugate gradient (primal)
      NEWTON: Newton (primal)
    """

    # unsupported: PGS
    CG = mujoco.mjtSolver.mjSOL_CG
    NEWTON = mujoco.mjtSolver.mjSOL_NEWTON


class EqType(enum.IntEnum):
    """Type of equality constraint.

    Attributes:
      CONNECT: connect two bodies at a point (ball joint)
      WELD: fix relative position and orientation of two bodies
      JOINT: couple the values of two scalar joints with cubic
      TENDON: couple the lengths of two tendons with cubic
    """

    CONNECT = mujoco.mjtEq.mjEQ_CONNECT
    WELD = mujoco.mjtEq.mjEQ_WELD
    JOINT = mujoco.mjtEq.mjEQ_JOINT
    TENDON = mujoco.mjtEq.mjEQ_TENDON
    # unsupported: DISTANCE


class WrapType(enum.IntEnum):
    """Type of tendon wrap object.

    Attributes:
      JOINT: constant moment arm
      PULLEY: pulley used to split tendon
      SITE: pass through site
      SPHERE: wrap around sphere
      CYLINDER: wrap around (infinite) cylinder
    """

    JOINT = mujoco.mjtWrap.mjWRAP_JOINT
    PULLEY = mujoco.mjtWrap.mjWRAP_PULLEY
    SITE = mujoco.mjtWrap.mjWRAP_SITE
    SPHERE = mujoco.mjtWrap.mjWRAP_SPHERE
    CYLINDER = mujoco.mjtWrap.mjWRAP_CYLINDER


class TrnType(enum.IntEnum):
    """Type of actuator transmission.

    Attributes:
      JOINT: force on joint
      JOINTINPARENT: force on joint, expressed in parent frame
      SITE: force on site
    """

    JOINT = mujoco.mjtTrn.mjTRN_JOINT
    JOINTINPARENT = mujoco.mjtTrn.mjTRN_JOINTINPARENT
    TENDON = mujoco.mjtTrn.mjTRN_TENDON
    SITE = mujoco.mjtTrn.mjTRN_SITE
    # unsupported: SLIDERCRANK, BODY


class DynType(enum.IntEnum):
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


class GainType(enum.IntEnum):
    """Type of actuator gain.

    Attributes:
      FIXED: fixed gain
      AFFINE: const + kp*length + kv*velocity
    """

    FIXED = mujoco.mjtGain.mjGAIN_FIXED
    AFFINE = mujoco.mjtGain.mjGAIN_AFFINE
    # unsupported: MUSCLE
    # unsupported: USER


class BiasType(enum.IntEnum):
    """Type of actuator bias.

    Attributes:
      NONE: no bias
      AFFINE: const + kp*length + kv*velocity
    """

    NONE = mujoco.mjtBias.mjBIAS_NONE
    AFFINE = mujoco.mjtBias.mjBIAS_AFFINE
    # unsupported: MUSCLE, USER


class ConstraintType(enum.IntEnum):
    """Type of constraint.

    Attributes:
      EQUALITY: equality constraint
      FRICTION_DOF: DOF friction
      FRICTION_TENDON: tendon friction
      LIMIT_JOINT: joint limit
      LIMIT_TENDON: tendon limit
      CONTACT_FRICTIONLESS: frictionless contact
      CONTACT_PYRAMIDAL: frictional contact, pyramidal friction cone
      CONTACT_ELLIPTIC: frictional contact, elliptic friction cone
    """

    EQUALITY = mujoco.mjtConstraint.mjCNSTR_EQUALITY
    FRICTION_DOF = mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    FRICTION_TENDON = mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON
    LIMIT_JOINT = mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT
    LIMIT_TENDON = mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
    CONTACT_FRICTIONLESS = mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS
    CONTACT_PYRAMIDAL = mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
    CONTACT_ELLIPTIC = mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC


class CamLightType(enum.IntEnum):
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


class SensorType(enum.IntEnum):
    """Type of sensor."""

    MAGNETOMETER = mujoco.mjtSensor.mjSENS_MAGNETOMETER
    CAMPROJECTION = mujoco.mjtSensor.mjSENS_CAMPROJECTION
    RANGEFINDER = mujoco.mjtSensor.mjSENS_RANGEFINDER
    JOINTPOS = mujoco.mjtSensor.mjSENS_JOINTPOS
    TENDONPOS = mujoco.mjtSensor.mjSENS_TENDONPOS
    ACTUATORPOS = mujoco.mjtSensor.mjSENS_ACTUATORPOS
    BALLQUAT = mujoco.mjtSensor.mjSENS_BALLQUAT
    FRAMEPOS = mujoco.mjtSensor.mjSENS_FRAMEPOS
    FRAMEXAXIS = mujoco.mjtSensor.mjSENS_FRAMEXAXIS
    FRAMEYAXIS = mujoco.mjtSensor.mjSENS_FRAMEYAXIS
    FRAMEZAXIS = mujoco.mjtSensor.mjSENS_FRAMEZAXIS
    FRAMEQUAT = mujoco.mjtSensor.mjSENS_FRAMEQUAT
    SUBTREECOM = mujoco.mjtSensor.mjSENS_SUBTREECOM
    CLOCK = mujoco.mjtSensor.mjSENS_CLOCK
    VELOCIMETER = mujoco.mjtSensor.mjSENS_VELOCIMETER
    GYRO = mujoco.mjtSensor.mjSENS_GYRO
    JOINTVEL = mujoco.mjtSensor.mjSENS_JOINTVEL
    TENDONVEL = mujoco.mjtSensor.mjSENS_TENDONVEL
    ACTUATORVEL = mujoco.mjtSensor.mjSENS_ACTUATORVEL
    BALLANGVEL = mujoco.mjtSensor.mjSENS_BALLANGVEL
    FRAMELINVEL = mujoco.mjtSensor.mjSENS_FRAMELINVEL
    FRAMEANGVEL = mujoco.mjtSensor.mjSENS_FRAMEANGVEL
    SUBTREELINVEL = mujoco.mjtSensor.mjSENS_SUBTREELINVEL
    SUBTREEANGMOM = mujoco.mjtSensor.mjSENS_SUBTREEANGMOM
    TOUCH = mujoco.mjtSensor.mjSENS_TOUCH
    CONTACT = mujoco.mjtSensor.mjSENS_CONTACT
    ACCELEROMETER = mujoco.mjtSensor.mjSENS_ACCELEROMETER
    FORCE = mujoco.mjtSensor.mjSENS_FORCE
    TORQUE = mujoco.mjtSensor.mjSENS_TORQUE
    ACTUATORFRC = mujoco.mjtSensor.mjSENS_ACTUATORFRC
    JOINTACTFRC = mujoco.mjtSensor.mjSENS_JOINTACTFRC
    TENDONACTFRC = mujoco.mjtSensor.mjSENS_TENDONACTFRC
    FRAMELINACC = mujoco.mjtSensor.mjSENS_FRAMELINACC
    FRAMEANGACC = mujoco.mjtSensor.mjSENS_FRAMEANGACC


class ObjType(enum.IntEnum):
    """Type of object.

    Attributes:
      UNKNOWN: unknown object type
      BODY: body
      XBODY: body, used to access regular frame instead of i-frame
      GEOM: geom
      SITE: site
      CAMERA: camera
    """

    UNKNOWN = mujoco.mjtObj.mjOBJ_UNKNOWN
    BODY = mujoco.mjtObj.mjOBJ_BODY
    XBODY = mujoco.mjtObj.mjOBJ_XBODY
    GEOM = mujoco.mjtObj.mjOBJ_GEOM
    SITE = mujoco.mjtObj.mjOBJ_SITE
    CAMERA = mujoco.mjtObj.mjOBJ_CAMERA


class Statistic(MjTensorClass):
    """Model statistics (in qpos0).

    Attributes:
      meaninertia: mean diagonal inertia
      meanmass: mean body mass
      meansize: mean body size
      extent: spatial extent
      center: center of model
    """

    meaninertia: float
    meanmass: torch.Tensor
    meansize: torch.Tensor
    extent: torch.Tensor
    center: torch.Tensor


class Option(MjTensorClass):
    """Physics options.

    Attributes:
      iterations:     number of main solver iterations
      ls_iterations:  maximum number of CG/Newton linesearch iterations
      tolerance:      main solver tolerance
      ls_tolerance:   CG/Newton linesearch tolerance
      impratio:       ratio of friction-to-normal constraint impedance
      gravity:        gravitational acceleration                         (3,)
      density:        density of medium
      viscosity:      viscosity of medium
      magnetic:       global magnetic flux                               (3,)
      wind:           wind (for lift, drag and viscosity)                 (3,)
      jacobian:       type of constraint Jacobian
      cone:           type of friction cone
      disableflags:   bit flags for disabling standard features
      enableflags:    bit flags for enabling optional features
      integrator:     integration mode
      solver:         solver algorithm
      timestep:       timestep
      o_margin:       override margin
      o_solref:       override solver reference
      o_solimp:       override solver impedance
      o_friction:     override friction
      disableactuator: disable actuator
      sdf_initpoints: number of SDF init points
      has_fluid_params: whether wind/density/viscosity are nonzero
    """

    iterations: int
    ls_iterations: int
    tolerance: float
    ls_tolerance: float
    impratio: torch.Tensor
    gravity: torch.Tensor
    density: torch.Tensor
    viscosity: torch.Tensor
    magnetic: torch.Tensor
    wind: torch.Tensor
    jacobian: JacobianType
    cone: ConeType
    disableflags: DisableBit
    enableflags: int
    integrator: IntegratorType
    solver: SolverType
    timestep: torch.Tensor
    # impl-specific fields (flattened from OptionJAX):
    o_margin: torch.Tensor
    o_solref: torch.Tensor
    o_solimp: torch.Tensor
    o_friction: torch.Tensor
    disableactuator: int
    sdf_initpoints: int
    has_fluid_params: bool


class Model(MjTensorClass):
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
      nlight: number of lights
      nmesh: number of meshes
      npair: number of predefined geom pairs
      nexclude: number of excluded geom pairs
      neq: number of equality constraints
      ntendon: number of tendons
      nwrap: number of wrap objects in all tendon paths
      nsensor: number of sensors
      nnumeric: number of numeric custom fields
      nmocap: number of mocap bodies
      nM: number of non-zeros in sparse inertia matrix
      nsensordata: number of elements in sensor data vector
      opt: physics options
      stat: model statistics
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
    nlight: int
    nmesh: int
    npair: int
    nexclude: int
    neq: int
    ntendon: int
    nwrap: int
    nsensor: int
    nnumeric: int
    nmocap: int
    nM: int  # pylint:disable=invalid-name
    nsensordata: int
    opt: Option
    stat: Statistic
    qpos0: torch.Tensor
    qpos_spring: torch.Tensor
    body_parentid: np.ndarray
    body_mocapid: np.ndarray
    body_rootid: np.ndarray
    body_weldid: np.ndarray
    body_jntnum: np.ndarray
    body_jntadr: np.ndarray
    body_sameframe: np.ndarray
    body_dofnum: np.ndarray
    body_dofadr: np.ndarray
    body_treeid: np.ndarray
    body_geomnum: np.ndarray
    body_geomadr: np.ndarray
    body_simple: np.ndarray
    body_pos: torch.Tensor
    body_quat: torch.Tensor
    body_ipos: torch.Tensor
    body_iquat: torch.Tensor
    body_mass: torch.Tensor
    body_subtreemass: torch.Tensor
    body_inertia: torch.Tensor
    body_gravcomp: torch.Tensor
    body_invweight0: torch.Tensor
    jnt_type: np.ndarray
    jnt_qposadr: np.ndarray
    jnt_dofadr: np.ndarray
    jnt_bodyid: np.ndarray
    jnt_group: np.ndarray
    jnt_limited: np.ndarray
    jnt_actfrclimited: np.ndarray
    jnt_actgravcomp: np.ndarray
    jnt_solref: torch.Tensor
    jnt_solimp: torch.Tensor
    jnt_pos: torch.Tensor
    jnt_axis: torch.Tensor
    jnt_stiffness: torch.Tensor
    jnt_range: torch.Tensor
    jnt_actfrcrange: torch.Tensor
    jnt_margin: torch.Tensor
    dof_bodyid: np.ndarray
    dof_jntid: np.ndarray
    dof_parentid: np.ndarray
    dof_treeid: np.ndarray
    dof_Madr: np.ndarray  # pylint:disable=invalid-name
    dof_simplenum: np.ndarray
    dof_solref: torch.Tensor
    dof_solimp: torch.Tensor
    dof_frictionloss: torch.Tensor
    dof_armature: torch.Tensor
    dof_damping: torch.Tensor
    dof_invweight0: torch.Tensor
    dof_M0: torch.Tensor  # pylint:disable=invalid-name
    geom_type: np.ndarray
    geom_contype: np.ndarray
    geom_conaffinity: np.ndarray
    geom_condim: np.ndarray
    geom_bodyid: np.ndarray
    geom_sameframe: np.ndarray
    geom_dataid: np.ndarray
    geom_group: np.ndarray
    geom_matid: torch.Tensor
    geom_priority: np.ndarray
    geom_solmix: torch.Tensor
    geom_solref: torch.Tensor
    geom_solimp: torch.Tensor
    geom_size: torch.Tensor
    geom_aabb: torch.Tensor
    geom_rbound: torch.Tensor
    geom_pos: torch.Tensor
    geom_quat: torch.Tensor
    geom_friction: torch.Tensor
    geom_margin: torch.Tensor
    geom_gap: torch.Tensor
    geom_fluid: np.ndarray
    geom_rgba: torch.Tensor
    site_type: np.ndarray
    site_bodyid: np.ndarray
    site_sameframe: np.ndarray
    site_size: np.ndarray
    site_pos: torch.Tensor
    site_quat: torch.Tensor
    cam_mode: np.ndarray
    cam_bodyid: np.ndarray
    cam_targetbodyid: np.ndarray
    cam_pos: torch.Tensor
    cam_quat: torch.Tensor
    cam_poscom0: torch.Tensor
    cam_pos0: torch.Tensor
    cam_mat0: torch.Tensor
    cam_fovy: np.ndarray
    cam_resolution: np.ndarray
    cam_sensorsize: np.ndarray
    cam_intrinsic: np.ndarray
    light_mode: np.ndarray
    light_bodyid: np.ndarray
    light_targetbodyid: np.ndarray
    light_type: torch.Tensor
    light_castshadow: torch.Tensor
    light_pos: torch.Tensor
    light_dir: torch.Tensor
    light_poscom0: torch.Tensor
    light_pos0: torch.Tensor
    light_dir0: torch.Tensor
    light_cutoff: torch.Tensor
    mesh_vertadr: np.ndarray
    mesh_vertnum: np.ndarray
    mesh_faceadr: np.ndarray
    mesh_normaladr: np.ndarray
    mesh_normalnum: np.ndarray
    mesh_graphadr: np.ndarray
    mesh_vert: np.ndarray
    mesh_normal: np.ndarray
    mesh_face: np.ndarray
    mesh_graph: np.ndarray
    mesh_pos: np.ndarray
    mesh_quat: np.ndarray
    mesh_texcoordadr: np.ndarray
    mesh_texcoordnum: np.ndarray
    mesh_texcoord: np.ndarray
    hfield_size: np.ndarray
    hfield_nrow: np.ndarray
    hfield_ncol: np.ndarray
    hfield_adr: np.ndarray
    hfield_data: torch.Tensor
    mat_rgba: torch.Tensor
    pair_dim: np.ndarray
    pair_geom1: np.ndarray
    pair_geom2: np.ndarray
    pair_signature: np.ndarray
    pair_solref: torch.Tensor
    pair_solreffriction: torch.Tensor
    pair_solimp: torch.Tensor
    pair_margin: torch.Tensor
    pair_gap: torch.Tensor
    pair_friction: torch.Tensor
    exclude_signature: np.ndarray
    eq_type: np.ndarray
    eq_obj1id: np.ndarray
    eq_obj2id: np.ndarray
    eq_objtype: np.ndarray
    eq_active0: np.ndarray
    eq_solref: torch.Tensor
    eq_solimp: torch.Tensor
    eq_data: torch.Tensor
    tendon_adr: np.ndarray
    tendon_num: np.ndarray
    tendon_limited: np.ndarray
    tendon_actfrclimited: np.ndarray
    tendon_solref_lim: torch.Tensor
    tendon_solimp_lim: torch.Tensor
    tendon_solref_fri: torch.Tensor
    tendon_solimp_fri: torch.Tensor
    tendon_range: torch.Tensor
    tendon_actfrcrange: torch.Tensor
    tendon_margin: torch.Tensor
    tendon_stiffness: torch.Tensor
    tendon_damping: torch.Tensor
    tendon_armature: torch.Tensor
    tendon_frictionloss: torch.Tensor
    tendon_lengthspring: torch.Tensor
    tendon_length0: torch.Tensor
    tendon_invweight0: torch.Tensor
    wrap_type: np.ndarray
    wrap_objid: np.ndarray
    wrap_prm: np.ndarray
    actuator_trntype: np.ndarray
    actuator_dyntype: np.ndarray
    actuator_gaintype: np.ndarray
    actuator_biastype: np.ndarray
    actuator_trnid: np.ndarray
    actuator_actadr: np.ndarray
    actuator_actnum: np.ndarray
    actuator_group: np.ndarray
    actuator_ctrllimited: np.ndarray
    actuator_forcelimited: np.ndarray
    actuator_actlimited: np.ndarray
    actuator_dynprm: torch.Tensor
    actuator_gainprm: torch.Tensor
    actuator_biasprm: torch.Tensor
    actuator_actearly: np.ndarray
    actuator_ctrlrange: torch.Tensor
    actuator_forcerange: torch.Tensor
    actuator_actrange: torch.Tensor
    actuator_gear: torch.Tensor
    actuator_cranklength: np.ndarray
    actuator_acc0: torch.Tensor
    actuator_lengthrange: np.ndarray
    sensor_type: np.ndarray
    sensor_datatype: np.ndarray
    sensor_needstage: np.ndarray
    sensor_objtype: np.ndarray
    sensor_objid: np.ndarray
    sensor_reftype: np.ndarray
    sensor_refid: np.ndarray
    sensor_intprm: np.ndarray
    sensor_dim: np.ndarray
    sensor_adr: np.ndarray
    sensor_cutoff: np.ndarray
    numeric_adr: np.ndarray
    numeric_data: np.ndarray
    key_time: np.ndarray
    key_qpos: np.ndarray
    key_qvel: np.ndarray
    key_act: np.ndarray
    key_mpos: np.ndarray
    key_mquat: np.ndarray
    key_ctrl: np.ndarray
    name_bodyadr: np.ndarray
    name_jntadr: np.ndarray
    name_geomadr: np.ndarray
    name_siteadr: np.ndarray
    name_camadr: np.ndarray
    name_meshadr: np.ndarray
    name_pairadr: np.ndarray
    name_eqadr: np.ndarray
    name_tendonadr: np.ndarray
    name_actuatoradr: np.ndarray
    name_sensoradr: np.ndarray
    name_numericadr: np.ndarray
    names: bytes
    # Torch-impl-specific fields (flattened from upstream ModelJAX):
    dof_hasfrictionloss: np.ndarray
    geom_rbound_hfield: np.ndarray
    geom_convex_face: tuple[torch.Tensor | None, ...]
    geom_convex_vert: tuple[torch.Tensor | None, ...]
    geom_convex_edge: tuple[torch.Tensor | None, ...]
    geom_convex_facenormal: tuple[torch.Tensor | None, ...]
    mesh_convex: tuple[ConvexMesh, ...]
    tendon_hasfrictionloss: np.ndarray
    has_gravcomp: bool
    dof_tri_row: np.ndarray
    dof_tri_col: np.ndarray
    actuator_info: tuple
    constraint_sizes_py: tuple
    condim_counts_py: tuple
    condim_tensor_py: torch.Tensor
    constraint_data_py: dict
    collision_groups_py: tuple
    collision_max_cp_py: int
    collision_total_contacts_py: int
    sensor_groups_pos_py: tuple
    sensor_groups_vel_py: tuple
    sensor_groups_acc_py: tuple
    sensor_disabled_py: bool
    cache_id: int
    # Pre-cached tensor versions of numpy model fields (auto-moved by .to())
    body_rootid_t: torch.Tensor
    dof_bodyid_t: torch.Tensor
    dof_Madr_t: torch.Tensor
    dof_tri_row_t: torch.Tensor
    dof_tri_col_t: torch.Tensor
    geom_bodyid_t: torch.Tensor
    site_bodyid_t: torch.Tensor
    dof_jntid_t: torch.Tensor
    actuator_ctrllimited_bool: torch.Tensor
    actuator_forcelimited_bool: torch.Tensor
    jnt_actfrclimited_bool: torch.Tensor
    actuator_actlimited_bool: torch.Tensor
    actuator_actadr_neg1: torch.Tensor
    # Pre-computed sparse mass matrix index pattern (used by solver, smooth)
    sparse_i_t: torch.Tensor
    sparse_j_t: torch.Tensor
    sparse_madr_t: torch.Tensor
    # Pre-computed factor_m indices
    factor_m_madr_ds_t: torch.Tensor
    factor_m_updates: tuple
    # Pre-computed solve_m indices
    solve_m_updates_j: tuple
    solve_m_updates_i: tuple


# Model.names collides with TensorDict.names property.  A __getattribute__
# override would fix this but causes graph breaks on EVERY attribute access,
# which makes torch.compile skip entire frames (e.g. _advance, _euler).
# Instead, shadow just the ``names`` property with a targeted descriptor
# that reads/writes the stored field value directly.
def _model_names_get(self):
    return self._tensordict["names"]


def _model_names_set(self, value):
    self._tensordict["names"] = value


Model.names = property(_model_names_get, _model_names_set)


class Contact(MjTensorClass):
    """Result of collision detection functions.

    Attributes:
      dist: distance between nearest points; neg: penetration
      pos: position of contact point: midpoint between geoms            (3,)
      frame: normal is in [0-2]                                         (3, 3)
      includemargin: include if dist<includemargin=margin-gap
      friction: tangent1, 2, spin, roll1, 2                             (5,)
      solref: constraint solver reference, normal direction             (mjNREF,)
      solreffriction: constraint solver reference, friction directions  (mjNREF,)
      solimp: constraint solver impedance                               (mjNIMP,)
      contact_dim: contact space dimensionality: 1, 3, 4 or 6
      geom1: id of geom 1
      geom2: id of geom 2
      geom: geom ids                                                    (2,)
      efc_address: address in efc; -1: not included
    """

    dist: torch.Tensor
    pos: torch.Tensor
    frame: torch.Tensor
    includemargin: torch.Tensor
    friction: torch.Tensor
    solref: torch.Tensor
    solreffriction: torch.Tensor
    solimp: torch.Tensor
    # unsupported: mu, H
    contact_dim: torch.Tensor
    geom1: torch.Tensor
    geom2: torch.Tensor
    geom: torch.Tensor
    efc_address: torch.Tensor

    @classmethod
    def zero(cls, shape=(0,), device=None) -> "Contact":
        """Returns a contact filled with zeros."""
        return Contact(
            dist=torch.zeros(shape, device=device),
            pos=torch.zeros(shape + (3,), device=device),
            frame=torch.zeros(shape + (3, 3), device=device),
            includemargin=torch.zeros(shape, device=device),
            friction=torch.zeros(shape + (5,), device=device),
            solref=torch.zeros(shape + (mujoco.mjNREF,), device=device),
            solreffriction=torch.zeros(shape + (mujoco.mjNREF,), device=device),
            solimp=torch.zeros(shape + (mujoco.mjNIMP,), device=device),
            contact_dim=torch.zeros(shape, dtype=torch.int32, device=device),
            geom1=torch.zeros(shape, dtype=torch.int32, device=device),
            geom2=torch.zeros(shape, dtype=torch.int32, device=device),
            geom=torch.zeros(shape + (2,), dtype=torch.int32, device=device),
            efc_address=torch.zeros(shape, dtype=torch.int32, device=device),
            batch_size=list(shape),
        )


class Data(MjTensorClass):
    """Dynamic state that updates each step.

    Attributes:
      time: simulation time
      qpos: position                                                (nq,)
      qvel: velocity                                                (nv,)
      act: actuator activation                                      (na,)
      qacc_warmstart: acceleration used for warmstart               (nv,)
      ctrl: control                                                 (nu,)
      qfrc_applied: applied generalized force                       (nv,)
      xfrc_applied: applied Cartesian force/torque                  (nbody, 6)
      eq_active: enable/disable constraints                         (neq,)
      mocap_pos: positions of mocap bodies                           (nmocap, 3)
      mocap_quat: orientations of mocap bodies                      (nmocap, 4)
      qacc: acceleration                                            (nv,)
      act_dot: time-derivative of actuator activation               (na,)
      userdata: user data                                           (nuserdata,)
      sensordata: sensor data                                       (nsensordata,)
      xpos: Cartesian position of body frame                        (nbody, 3)
      xquat: Cartesian orientation of body frame                    (nbody, 4)
      xmat: Cartesian orientation of body frame                     (nbody, 3, 3)
      xipos: Cartesian position of body com                         (nbody, 3)
      ximat: Cartesian orientation of body inertia                  (nbody, 3, 3)
      xanchor: Cartesian position of joint anchor                   (njnt, 3)
      xaxis: Cartesian joint axis                                   (njnt, 3)
      ten_length: tendon lengths                                    (ntendon,)
      geom_xpos: Cartesian geom position                            (ngeom, 3)
      geom_xmat: Cartesian geom orientation                         (ngeom, 3, 3)
      site_xpos: Cartesian site position                            (nsite, 3)
      site_xmat: Cartesian site orientation                         (nsite, 3, 3)
      cam_xpos: camera positions                                    (ncam, 3)
      cam_xmat: camera rotation matrices                            (ncam, 3, 3)
      subtree_com: center of mass of each subtree                   (nbody, 3)
      cdof: com-based motion axis of each dof                       (nv, 6)
      cinert: com-based body inertia and mass                       (nbody, 10)
      actuator_length: actuator lengths                             (nu,)
      actuator_moment: actuator moments                             (nu, nv)
      crb: com-based composite inertia and mass                     (nbody, 10)
      qM: total inertia (sparse)                                    (nM,)
      qLD: L'*D*L factorization of M (sparse)                       (nM,)
      qLDiagInv: 1/diag(D)                                          (nv,)
      ten_wrapadr: tendon wrap addresses                            (ntendon,)
      ten_wrapnum: tendon wrap counts                               (ntendon,)
      ten_J: tendon Jacobian                                        (ntendon, nv)
      ten_velocity: tendon velocities                               (ntendon,)
      wrap_obj: wrap object data                                    (nwrap, 2)
      wrap_xpos: wrap positions                                     (nwrap, 6)
      contact: list of all detected contacts                        (ncon,)
      efc_type: constraint type                                     (nefc,)
      efc_J: constraint Jacobian                                    (nefc, nv)
      efc_pos: constraint position error                            (nefc,)
      efc_margin: constraint margin                                 (nefc,)
      efc_frictionloss: frictionloss (friction)                     (nefc,)
      efc_D: constraint mass                                        (nefc,)
      efc_aref: reference pseudo-acceleration                       (nefc,)
      efc_force: constraint force in constraint space               (nefc,)
      actuator_velocity: actuator velocities                        (nu,)
      cvel: com-based velocity [3D rot; 3D tran]                    (nbody, 6)
      cdof_dot: time-derivative of cdof                             (nv, 6)
      qfrc_bias: C(qpos,qvel)                                       (nv,)
      qfrc_gravcomp: gravity compensation force                     (nv,)
      qfrc_fluid: fluid forces                                      (nv,)
      qfrc_passive: passive force                                   (nv,)
      actuator_force: actuator force in actuation space             (nu,)
      qfrc_actuator: actuator force                                 (nv,)
      qfrc_smooth: net unconstrained force                          (nv,)
      qacc_smooth: unconstrained acceleration                       (nv,)
      qfrc_constraint: constraint force                             (nv,)
      qfrc_inverse: net external force; should equal:               (nv,)
        qfrc_applied + J'*xfrc_applied + qfrc_actuator
      cacc: com-based acceleration                                  (nbody, 6)
      cfrc_int: internal com-based force                            (nbody, 6)
      cfrc_ext: external com-based force                            (nbody, 6)
      subtree_linvel: subtree linear velocity                       (nbody, 3)
      subtree_angmom: subtree angular momentum                      (nbody, 3)
    """

    # solver statistics:
    solver_niter: torch.Tensor
    # sizes (variable in MJ, constant in MJX).
    # Stored as 0-d int32 tensors so that torch.vmap can batch/unbatch them.
    ne: torch.Tensor
    nf: torch.Tensor
    nl: torch.Tensor
    nefc: torch.Tensor
    ncon: torch.Tensor
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
    # mocap data:
    mocap_pos: torch.Tensor
    mocap_quat: torch.Tensor
    # dynamics:
    qacc: torch.Tensor
    act_dot: torch.Tensor
    # user data:
    userdata: torch.Tensor
    sensordata: torch.Tensor
    # position dependent:
    xpos: torch.Tensor
    xquat: torch.Tensor
    xmat: torch.Tensor
    xipos: torch.Tensor
    ximat: torch.Tensor
    xanchor: torch.Tensor
    xaxis: torch.Tensor
    ten_length: torch.Tensor
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
    ten_wrapadr: torch.Tensor
    ten_wrapnum: torch.Tensor
    ten_J: torch.Tensor  # pylint:disable=invalid-name
    ten_velocity: torch.Tensor
    wrap_obj: torch.Tensor
    wrap_xpos: torch.Tensor
    contact: Contact
    efc_type: torch.Tensor
    efc_J: torch.Tensor  # pylint:disable=invalid-name
    efc_pos: torch.Tensor
    efc_margin: torch.Tensor
    efc_frictionloss: torch.Tensor
    efc_D: torch.Tensor  # pylint:disable=invalid-name
    efc_aref: torch.Tensor
    efc_force: torch.Tensor
    # position, velocity dependent:
    actuator_velocity: torch.Tensor
    cvel: torch.Tensor
    cdof_dot: torch.Tensor
    qfrc_bias: torch.Tensor
    qfrc_gravcomp: torch.Tensor
    qfrc_fluid: torch.Tensor
    qfrc_passive: torch.Tensor
    # position, velocity, control & acceleration dependent:
    actuator_force: torch.Tensor
    qfrc_actuator: torch.Tensor
    qfrc_smooth: torch.Tensor
    qacc_smooth: torch.Tensor
    qfrc_constraint: torch.Tensor
    qfrc_inverse: torch.Tensor
    # post-constraint:
    cacc: torch.Tensor
    cfrc_int: torch.Tensor
    cfrc_ext: torch.Tensor
    subtree_linvel: torch.Tensor
    subtree_angmom: torch.Tensor
