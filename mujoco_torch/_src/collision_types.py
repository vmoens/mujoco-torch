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
"""Collision base types."""

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
from mujoco_torch._src.dataclasses import MjTensorClass  # pylint: disable=g-importing-member

# Collision returned by collision functions:
# - distance distance between nearest points; neg: penetration
# - position (3,) position of contact point: midpoint between geoms
# - frame (3, 3) normal is in [0, :], points from geom[0] to geom[1]
Collision = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class GeomInfo(MjTensorClass):
  """Geom properties for primitive and convex shapes."""

  pos: torch.Tensor
  mat: torch.Tensor
  geom_size: torch.Tensor
  face: Optional[torch.Tensor] = None
  vert: Optional[torch.Tensor] = None
  edge: Optional[torch.Tensor] = None
  facenorm: Optional[torch.Tensor] = None


class ConvexInfo(MjTensorClass):
  """Geom properties for convex meshes."""

  pos: torch.Tensor
  mat: torch.Tensor
  geom_size: torch.Tensor
  vert: torch.Tensor
  face: torch.Tensor
  face_normal: torch.Tensor
  edge: torch.Tensor
  edge_face_normal: torch.Tensor

  @property
  def facenorm(self) -> torch.Tensor:
    """Alias for face_normal for collision API compatibility."""
    return self.face_normal


class HFieldInfo(MjTensorClass):
  """Geom properties for height fields."""

  pos: torch.Tensor
  mat: torch.Tensor
  hfield_size: np.ndarray
  nrow: int
  ncol: int
  hfield_data: torch.Tensor


@dataclasses.dataclass(frozen=True)
class FunctionKey:
  """Specifies how geom pairs group into collision_driver's function table.

  Attributes:
    types: geom type pair, which determines the collision function
    data_ids: geom data id pair: mesh id for mesh geoms, otherwise -1. Meshes
      have distinct face/vertex counts, so must occupy distinct entries in the
      collision function table.
    condim: grouping by condim of the collision ensures that the size of the
      resulting constraint jacobian is determined at compile time.
    subgrid_size: the size determines the hfield subgrid to collide with
  """

  types: Tuple[int, int]
  data_ids: Tuple[int, int]
  condim: int
  subgrid_size: Tuple[int, int] = (-1, -1)
