from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ModelConfig
from ..domain import FEMInput


@dataclass
class WallBaseCheckUnit:
    wall_id: int
    base_nodes: list[int]
    length: float
    thickness: float


@dataclass
class ModelBuildResult:
    master_nodes: list[int]
    floor_area: float
    wall_base_units: list[WallBaseCheckUnit]
    floor_story_nodes: list[list[int]]


class StructuralModelBuilder(ABC):
    name: str

    @abstractmethod
    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        raise NotImplementedError
