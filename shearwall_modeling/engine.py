from dataclasses import dataclass
from typing import Union

import openseespy.opensees as ops

from .analysis.response_spectrum import run_builtin_rsa
from .builders.registry import DEFAULT_REGISTRY, BuilderRegistry
from .config import ModelConfig
from .domain import FEMInput


@dataclass
class AnalysisResult:
    drifts: dict[str, list[float]]


class ShearWallAnalysisEngine:
    def __init__(self, registry: Union[BuilderRegistry, None] = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY

    def analyze(
        self, input_data: FEMInput, config: ModelConfig, builder_name: str = "detailed_shell"
    ) -> AnalysisResult:
        builder = self.registry.get(builder_name)
        build_result = builder.build(input_data, config)
        drifts = run_builtin_rsa(build_result.master_nodes, config)
        return AnalysisResult(drifts=drifts)

    @staticmethod
    def cleanup() -> None:
        ops.wipe()
