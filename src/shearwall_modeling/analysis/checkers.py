from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.constants import MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY
from .results import DirectionCheckResult, DirectionResponse, StoryMetric


class DirectionChecker(ABC):
    @abstractmethod
    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class TorsionChecker(DirectionChecker):
    limit_ratio: float = 1.5

    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        result.is_torsion_passed = all(metric.torsion_ratio <= self.limit_ratio for metric in response.metrics)


@dataclass(frozen=True)
class ShearWeightRatioChecker(DirectionChecker):
    min_ratio: float

    @classmethod
    def from_intensity(cls, intensity: float) -> "ShearWeightRatioChecker":
        key = round(float(intensity), 2)
        return cls(min_ratio=MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY.get(key, 0.016))

    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        result.is_shear_weight_passed = all(
            metric.shear_weight_ratio >= self.min_ratio for metric in response.metrics
        )


@dataclass(frozen=True)
class StiffnessChecker(DirectionChecker):
    adjacent_limit: float = 0.7
    average_limit: float = 0.8

    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        result.is_stiffness_passed = all(
            metric.stiffness_ratio_adjacent >= self.adjacent_limit
            and metric.stiffness_ratio_average >= self.average_limit
            for metric in response.metrics
        )


@dataclass(frozen=True)
class PeriodRatioChecker(DirectionChecker):
    limit_ratio: float = 0.9

    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        result.is_period_ratio_passed = response.modal_summary.is_period_ratio_passed


@dataclass(frozen=True)
class InterstoryDriftChecker(DirectionChecker):
    drift_limit: float

    def apply(self, response: DirectionResponse, result: DirectionCheckResult) -> None:
        max_metric = max(response.metrics, key=lambda metric: metric.drift_max)
        result.max_interstory_drift_ratio = max_metric.drift_max
        result.max_interstory_drift_story = max_metric.story
        result.interstory_drift_limit = self.drift_limit
        result.is_interstory_drift_passed = max_metric.drift_max <= self.drift_limit


def build_direction_result(response: DirectionResponse) -> DirectionCheckResult:
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


def update_story_stiffness_ratios(metrics: list[StoryMetric], stiffness_values: list[float]) -> None:
    for story_index, metric in enumerate(metrics):
        current_stiffness = stiffness_values[story_index]
        next_stiffness = stiffness_values[story_index + 1] if story_index < len(metrics) - 1 else current_stiffness
        upper_stiffness = [
            stiffness_values[story_index + offset]
            for offset in range(1, 4)
            if story_index + offset < len(metrics)
        ]
        average_upper_stiffness = (
            sum(upper_stiffness) / len(upper_stiffness) if upper_stiffness else current_stiffness
        )
        metric.stiffness_ratio_adjacent = current_stiffness / next_stiffness if next_stiffness > 1.0e-9 else 1.0
        metric.stiffness_ratio_average = (
            current_stiffness / average_upper_stiffness if average_upper_stiffness > 1.0e-9 else 1.0
        )
