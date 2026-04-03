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
    translational_mode_index: int | None
    translational_period: float | None
    torsional_mode_index: int | None
    torsional_period: float | None
    period_ratio: float | None
    is_period_ratio_passed: bool


@dataclass
class DirectionResponse:
    direction: str
    modal_periods: list[float]
    modal_summary: ModalSummary
    metrics: list[StoryMetric]


@dataclass
class DirectionCheckResult:
    direction: str
    metrics: list[StoryMetric]
    modal_periods: list[float]
    translational_mode_index: int | None
    translational_period: float | None
    torsional_mode_index: int | None
    torsional_period: float | None
    period_ratio: float | None
    is_torsion_passed: bool
    is_shear_weight_passed: bool
    is_stiffness_passed: bool
    is_period_ratio_passed: bool
    max_interstory_drift_ratio: float
    max_interstory_drift_story: int
    interstory_drift_limit: float
    is_interstory_drift_passed: bool


@dataclass
class WallAxialMetric:
    wall_id: int
    axial_force_n: float
    area_m2: float
    axial_stress_mpa: float
    axial_ratio: float
    ratio_limit: float
    is_passed: bool


def init_direction_check_result(response: DirectionResponse) -> DirectionCheckResult:
    summary = response.modal_summary
    return DirectionCheckResult(
        direction=response.direction,
        metrics=response.metrics,
        modal_periods=response.modal_periods.copy(),
        translational_mode_index=summary.translational_mode_index,
        translational_period=summary.translational_period,
        torsional_mode_index=summary.torsional_mode_index,
        torsional_period=summary.torsional_period,
        period_ratio=summary.period_ratio,
        is_torsion_passed=False,
        is_shear_weight_passed=False,
        is_stiffness_passed=False,
        is_period_ratio_passed=False,
        max_interstory_drift_ratio=0.0,
        max_interstory_drift_story=0,
        interstory_drift_limit=0.0,
        is_interstory_drift_passed=False,
    )
