from dataclasses import dataclass


@dataclass
class StoryMetric:
    story: int
    drift_max: float
    drift_avg: float
    drift_center: float
    torsion_ratio: float
    shear_weight_ratio: float
    stiffness_k: float
    stiffness_ratio_adjacent: float
    stiffness_ratio_average: float

    @property
    def gamma1(self) -> float:
        return self.stiffness_ratio_adjacent

    @gamma1.setter
    def gamma1(self, value: float) -> None:
        self.stiffness_ratio_adjacent = value

    @property
    def gamma2(self) -> float:
        return self.stiffness_ratio_average

    @gamma2.setter
    def gamma2(self, value: float) -> None:
        self.stiffness_ratio_average = value


@dataclass
class ModalSummary:
    periods: list[float]
    translational_mode_index: int | None
    translational_period: float | None
    torsional_mode_index: int | None
    torsional_period: float | None
    period_ratio: float | None


@dataclass
class DirectionResponse:
    direction: str
    metrics: list[StoryMetric]


@dataclass
class DirectionCheckResult:
    direction: str
    metrics: list[StoryMetric]
    is_torsion_passed: bool
    is_shear_weight_passed: bool
    is_stiffness_passed: bool
    is_period_ratio_passed: bool
    max_interstory_drift_ratio: float
    max_interstory_drift_story: int
    interstory_drift_limit: float
    is_interstory_drift_passed: bool


@dataclass
class OverallCheckResult:
    # direction-specific
    is_torsion_passed: bool = False
    is_shear_weight_passed: bool = False
    is_stiffness_passed: bool = False
    is_interstory_drift_passed: bool = False
    # not direction-specific
    is_period_ratio_passed: bool = False


@dataclass
class WallAxialMetric:
    wall_id: int
    axial_force_n: float
    area_m2: float
    axial_stress_mpa: float
    axial_ratio: float
    ratio_limit: float
    is_passed: bool


@dataclass
class AnalysisResult:
    eigen_values: list[float]
    modal_periods: list[float]
    modal_summary: ModalSummary
    story_weights: list[float]
    direction_responses: dict[str, DirectionResponse]
    gravity_beam_forces: dict[tuple[int, int], tuple[float, float, float]]
    gravity_wall_forces: dict[tuple[int, int], tuple[float, float, float]]
    seismic_beam_forces: dict[tuple[int, int], tuple[float, float, float]]
    seismic_wall_forces: dict[tuple[int, int], tuple[float, float, float, float]]
    wall_axial_metrics: list[WallAxialMetric]


@dataclass
class ResponseSpectrumCaseResult:
    eigen_values: list[float]
    modal_periods: list[float]
    modal_summary: ModalSummary
    story_weights: list[float]
    direction_responses: dict[str, DirectionResponse]
    seismic_beam_forces: dict[tuple[int, int], tuple[float, float, float]]
    seismic_wall_forces: dict[tuple[int, int], tuple[float, float, float, float]]


@dataclass
class GravityCaseResult:
    gravity_beam_forces: dict[tuple[int, int], tuple[float, float, float]]
    gravity_wall_forces: dict[tuple[int, int], tuple[float, float, float]]
    wall_axial_metrics: list[WallAxialMetric]


def init_direction_check_result(response: DirectionResponse) -> DirectionCheckResult:
    return DirectionCheckResult(
        direction=response.direction,
        metrics=response.metrics,
        is_torsion_passed=False,
        is_shear_weight_passed=False,
        is_stiffness_passed=False,
        is_period_ratio_passed=False,
        max_interstory_drift_ratio=0.0,
        max_interstory_drift_story=0,
        interstory_drift_limit=0.0,
        is_interstory_drift_passed=False,
    )
