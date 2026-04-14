from .base import StructuralModelBuilder
from .detailed_shell import DetailedShellBuilder
from .equi_frame import EquivalentFrameBuilder
from .mvlem_frame import MVLEMFrameBuilder


class BuilderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, StructuralModelBuilder] = {}

    def register(self, builder: StructuralModelBuilder) -> None:
        self._builders[builder.name] = builder

    def get(self, name: str) -> StructuralModelBuilder:
        if name not in self._builders:
            supported = ", ".join(sorted(self._builders.keys()))
            raise ValueError(f"Unsupported builder '{name}'. Supported: {supported}")
        return self._builders[name]

    def supported(self) -> list[str]:
        return sorted(self._builders.keys())


DEFAULT_REGISTRY = BuilderRegistry()
DEFAULT_REGISTRY.register(DetailedShellBuilder(None))
DEFAULT_REGISTRY.register(EquivalentFrameBuilder(None))
DEFAULT_REGISTRY.register(MVLEMFrameBuilder(None))
