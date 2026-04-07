from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.config import ModelConfig
from ..core.domain import BeamRole, FEMInput, PlanMember


@dataclass
class WallBaseCheckUnit:
    wall_id: int
    base_nodes: list[int]
    length: float
    thickness: float


@dataclass
class BeamElementUnit:
    beam_id: int
    story: int
    element_tag: int
    member: PlanMember
    role: BeamRole
    width: float
    depth: float
    length: float


@dataclass
class WallStoryElementUnit:
    wall_id: int
    story: int
    element_tags: list[int]
    bottom_nodes: list[int]
    top_nodes: list[int]
    node_coords: dict[int, tuple[float, float, float]]
    member: PlanMember
    thickness: float
    story_height: float


@dataclass
class ModelBuildResult:
    master_nodes: list[int]
    floor_area: float
    wall_base_units: list[WallBaseCheckUnit]
    floor_story_nodes: list[list[int]]
    beam_element_units: list[BeamElementUnit]
    wall_story_element_units: list[WallStoryElementUnit]


class StructuralModelBuilder(ABC):
    name: str

    @abstractmethod
    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        raise NotImplementedError
