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

from mujoco_torch._src import collision_driver, mesh, types
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
    }
)


def _device_put_torch(x):
    if x is None:
        return
    try:
        # Python scalar floats: use float64 to match DEFAULT_DTYPE precision
        if isinstance(x, float):
            return torch.tensor(x, dtype=torch.float64)
        # Use torch.tensor() to copy data (torch.as_tensor shares memory
        # with numpy arrays, which can be mutated by MuJoCo in-place)
        return torch.tensor(x)
    except (RuntimeError, TypeError):
        return torch.empty(x.shape if isinstance(x, np.ndarray) else len(x))


torch.device_put = _device_put_torch


def _model_derived(value: mujoco.MjModel) -> dict[str, Any]:
    mesh_kwargs = mesh.get(value)
    result = {}
    for k, v in mesh_kwargs.items():
        result[k] = tuple(torch.tensor(x) if x is not None else None for x in v)
    # mesh_convex: one ConvexMesh per mesh
    result["mesh_convex"] = tuple(mesh.convex(value, i) for i in range(value.nmesh))
    result["dof_hasfrictionloss"] = np.array(value.dof_frictionloss > 0)
    result["geom_rbound_hfield"] = np.array(value.geom_rbound)
    result["tendon_hasfrictionloss"] = np.array(value.tendon_frictionloss > 0)
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
    has_fluid = value.density > 0 or value.viscosity > 0 or (value.wind != 0.0).any()
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
def device_put(value: mujoco.MjData) -> types.Data: ...


@overload
def device_put(value: mujoco.MjModel) -> types.Model: ...


def device_put(value):
    """Places mujoco data onto a device.

    Args:
      value: a mujoco struct to transfer

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
            field_value = device_put(field_value)

        init_kwargs[f.name] = copy.copy(field_value)

    derived_kwargs = {}
    if isinstance(value, mujoco.MjModel):
        derived_kwargs = _model_derived(value)
    elif isinstance(value, mujoco.MjData):
        derived_kwargs = _data_derived(value)
    elif isinstance(value, mujoco.MjOption):
        derived_kwargs = _option_derived(value)

    if issubclass(clz, MjTensorClass):
        init_kwargs["batch_size"] = []
    return clz(**init_kwargs, **derived_kwargs)  # type: ignore


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
