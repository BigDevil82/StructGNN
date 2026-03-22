from dataclasses import dataclass, field


@dataclass
class MaterialConfig:
    E: float = 3.0e10
    G: float = 1.2e10


@dataclass
class SectionConfig:
    wall_thickness: float = 0.2
    beam_width: float = 0.3
    beam_depth: float = 0.5


@dataclass
class SeismicConfig:
    periods: list[float] = field(default_factory=list)
    sa: list[float] = field(default_factory=list)
    damping_ratio: float = 0.05
    combination_method: str = "CQC"

    def __post_init__(self) -> None:
        if not self.periods:
            self.periods = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 6.0]
        if not self.sa:
            g = 9.81
            self.sa = [0.4 * g, 1.0 * g, 1.0 * g, 0.8 * g, 0.4 * g, 0.2 * g, 0.1 * g]


@dataclass
class MassSourceConfig:
    dead_kpa: float = 5.0
    live_kpa: float = 2.0
    dead_factor: float = 1.0
    live_factor: float = 0.5
    gravity: float = 9.81

    def additional_mass_per_area(self) -> float:
        # Convert ETABS-like source loads (kN/m^2) to kg/m^2.
        source_kpa = self.dead_factor * self.dead_kpa + self.live_factor * self.live_kpa
        return source_kpa * 1000.0 / self.gravity


@dataclass
class ModelConfig:
    num_stories: int = 8
    story_height: float = 3.0
    mass_per_area: float = 1000.0
    mass_source: MassSourceConfig = field(default_factory=MassSourceConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    section: SectionConfig = field(default_factory=SectionConfig)
    seismic: SeismicConfig = field(default_factory=SeismicConfig)
    num_modes: int = 6
