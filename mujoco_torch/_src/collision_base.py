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
"""Collision base."""

import dataclasses
from typing import Dict, List, Optional, Tuple

import torch
from tensordict import tensorclass

# pylint: disable=g-importing-member
from mujoco_torch._src.dataclasses import PyTreeNode
from mujoco_torch._src.types import GeomType
import tensordict
# pylint: enable=g-importing-member

Contact = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclasses.dataclass
class Candidate:
  geom1: int
  geom2: int
  ipair: int
  geomp: int  # priority geom
  dim: int


CandidateSet = Dict[
    Tuple[GeomType, GeomType, Tuple[int, ...], Tuple[int, ...]],
    List[Candidate],
]

@tensordict.tensorclass(autocast=True)
class GeomInfo:
  """Collision info for a geom."""

  geom_id: torch.Tensor
  pos: torch.Tensor
  mat: torch.Tensor
  geom_size: torch.Tensor
  face: torch.Tensor = None
  vert: torch.Tensor = None
  edge_dir: torch.Tensor = None
  facenorm: torch.Tensor = None
  edge: torch.Tensor = None
  edge_face_normal: torch.Tensor = None

@tensordict.tensorclass(autocast=True)
class SolverParams:
  """Contact solver params."""

  friction: torch.Tensor
  solref: torch.Tensor
  solreffriction: torch.Tensor
  solimp: torch.Tensor
  margin: torch.Tensor
  gap: torch.Tensor
