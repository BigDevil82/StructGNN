from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..config import ModelConfig
from ..domain import FEMInput


@dataclass
class ModelBuildResult:
    master_nodes: list[int]
    floor_area: float


class StructuralModelBuilder(ABC):
    name: str

    @abstractmethod
    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        raise NotImplementedError
