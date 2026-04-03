import logging
from dataclasses import dataclass
from time import time
from typing import Any

import openseespy.opensees as ops

from ..builders.base import ModelBuildResult
from ..core.config import ModelConfig
from .checkers import update_story_stiffness_ratios
from .combinations import cqc, srss
from .modal import identify_dominant_modes, modal_periods_from_eigenvalues
from .results import DirectionResponse, ModalSummary, StoryMetric


@dataclass(frozen=True)
class AnalysisModelContext:
    build_result: ModelBuildResult
    config: ModelConfig
    logger: logging.Logger

    @property
    def master_nodes(self) -> list[int]:
        return self.build_result.master_nodes

    @property
    def floor_area(self) -> float:
        return self.build_result.floor_area

    @property
    def wall_base_units(self):
        return self.build_result.wall_base_units

    @property
    def floor_story_nodes(self) -> list[list[int]]:
        return self.build_result.floor_story_nodes

    @property
    def num_stories(self) -> int:
        return len(self.master_nodes)

    @property
    def story_heights(self) -> list[float]:
        return self.config.get_story_heights()

    @property
    def story_profiles(self):
        return self.config.resolve_story_profiles()

    @property
    def floor_masses(self) -> list[float]:
        return [ops.nodeMass(node, 1) for node in self.master_nodes]

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        nodes = ops.getNodeTags()
        xs = [ops.nodeCoord(node, 1) for node in nodes]
        ys = [ops.nodeCoord(node, 2) for node in nodes]
        return min(xs), max(xs), min(ys), max(ys)


class ResponseSpectrumAnalyzer:
    def __init__(self, context: AnalysisModelContext):
        self.context = context
        self.xmin, self.xmax, self.ymin, self.ymax = context.bbox

    def _combine(self, modal_values: list[float], eigen_values: list[float]) -> float:
        num_modes = len(modal_values)
        damping = [self.context.config.seismic.damping_ratio] * num_modes
        scale_factors = [1.0] * num_modes
        if self.context.config.seismic.combination_method.upper() == "CQC":
            return cqc(modal_values, eigen_values, damping, scale_factors)
        return srss(modal_values, scale_factors)

    def extract_modal_summary(self, modal_periods: list[float], modal_props: dict[str, Any]) -> ModalSummary:
        self.context.logger.info("\n[校核器] 正在根据模态参与质量比识别主导模态...")
        translational_mode_index, torsional_mode_index = identify_dominant_modes(modal_props)
        translational_period = (
            modal_periods[translational_mode_index - 1]
            if translational_mode_index is not None and translational_mode_index - 1 < len(modal_periods)
            else None
        )
        torsional_period = (
            modal_periods[torsional_mode_index - 1]
            if torsional_mode_index is not None and torsional_mode_index - 1 < len(modal_periods)
            else None
        )
        period_ratio = None
        is_period_ratio_passed = False
        if translational_period and torsional_period:
            period_ratio = torsional_period / translational_period
            is_period_ratio_passed = period_ratio < 0.9
        return ModalSummary(
            translational_mode_index=translational_mode_index,
            translational_period=translational_period,
            torsional_mode_index=torsional_mode_index,
            torsional_period=torsional_period,
            period_ratio=period_ratio,
            is_period_ratio_passed=is_period_ratio_passed,
        )

    def compute_story_weights(self) -> list[float]:
        story_weights = [0.0] * self.context.num_stories
        cumulative_mass = 0.0
        for story_index in reversed(range(self.context.num_stories)):
            cumulative_mass += self.context.floor_masses[story_index]
            story_weights[story_index] = cumulative_mass * self.context.config.seismic.gravity
        return story_weights

    def _compute_corner_displacements(
        self, dir_idx: int, center_disp: float, rotation_z: float, center_x: float, center_y: float
    ) -> list[float]:
        dx = [self.xmin - center_x, self.xmax - center_x, self.xmax - center_x, self.xmin - center_x]
        dy = [self.ymin - center_y, self.ymin - center_y, self.ymax - center_y, self.ymax - center_y]
        if dir_idx == 1:
            return [center_disp - rotation_z * delta_y for delta_y in dy]
        return [center_disp + rotation_z * delta_x for delta_x in dx]

    def _analyze_mode(
        self,
        dir_idx: int,
        eigen_value: float,
        mode_index: int,
        modal_drift_center: list[list[float]],
        modal_drift_corners: list[list[list[float]]],
        modal_story_shear: list[list[float]],
    ) -> None:
        ops.responseSpectrumAnalysis(
            dir_idx,
            "-Tn",
            *self.context.config.seismic.periods,
            "-Sa",
            *self.context.config.seismic.spectral_accel,
            "-mode",
            mode_index,
        )

        prev_center_disp = 0.0
        prev_corner_displacements = [0.0] * 4
        inertial_forces = [0.0] * self.context.num_stories
        for story_index, master in enumerate(self.context.master_nodes):
            center_disp = ops.nodeDisp(master, dir_idx)
            rotation_z = ops.nodeDisp(master, 6)
            center_x, center_y = ops.nodeCoord(master, 1), ops.nodeCoord(master, 2)
            inertial_forces[story_index] = self.context.floor_masses[story_index] * eigen_value * center_disp
            corner_displacements = self._compute_corner_displacements(
                dir_idx, center_disp, rotation_z, center_x, center_y
            )
            story_height = self.context.story_heights[story_index]
            modal_drift_center[story_index].append((center_disp - prev_center_disp) / story_height)
            for corner_index, corner_disp in enumerate(corner_displacements):
                modal_drift_corners[corner_index][story_index].append(
                    (corner_disp - prev_corner_displacements[corner_index]) / story_height
                )
            prev_center_disp = center_disp
            prev_corner_displacements = corner_displacements

        cumulative_shear = 0.0
        for story_index in reversed(range(self.context.num_stories)):
            cumulative_shear += inertial_forces[story_index]
            modal_story_shear[story_index].append(cumulative_shear)

    def analyze_direction(
        self,
        dir_idx: int,
        dir_name: str,
        eigen_values: list[float],
        modal_periods: list[float],
        modal_summary: ModalSummary,
        story_weights: list[float],
    ) -> DirectionResponse:
        modal_drift_center = [[] for _ in range(self.context.num_stories)]
        modal_drift_corners = [[[] for _ in range(self.context.num_stories)] for _ in range(4)]
        modal_story_shear = [[] for _ in range(self.context.num_stories)]

        for mode_index, eigen_value in enumerate(eigen_values, start=1):
            self._analyze_mode(
                dir_idx=dir_idx,
                eigen_value=eigen_value,
                mode_index=mode_index,
                modal_drift_center=modal_drift_center,
                modal_drift_corners=modal_drift_corners,
                modal_story_shear=modal_story_shear,
            )

        metrics: list[StoryMetric] = []
        combined_center_drifts: list[float] = []
        stiffness_values = [0.0] * self.context.num_stories
        for story_index in range(self.context.num_stories):
            story_shear = self._combine(modal_story_shear[story_index], eigen_values)
            center_drift = abs(self._combine(modal_drift_center[story_index], eigen_values))
            combined_center_drifts.append(center_drift)
            corner_drifts = [
                abs(self._combine(modal_drift_corners[corner_index][story_index], eigen_values))
                for corner_index in range(4)
            ]
            drift_max = max(corner_drifts)
            drift_min = min(corner_drifts)
            drift_avg = 0.5 * (drift_max + drift_min)
            torsion_ratio = drift_max / drift_avg if drift_avg > 1.0e-9 else 1.0
            shear_weight_ratio = (
                story_shear / story_weights[story_index] if story_weights[story_index] > 1.0e-9 else 0.0
            )
            stiffness = story_shear / center_drift if center_drift > 1.0e-9 else 0.0
            stiffness_values[story_index] = stiffness
            metrics.append(
                StoryMetric(
                    story=story_index + 1,
                    drift_max=drift_max,
                    drift_avg=drift_avg,
                    torsion_ratio=torsion_ratio,
                    shear_weight_ratio=shear_weight_ratio,
                    stiffness_k=stiffness,
                    stiffness_ratio_adjacent=1.0,
                    stiffness_ratio_average=1.0,
                )
            )

        update_story_stiffness_ratios(metrics, stiffness_values)
        return DirectionResponse(
            direction=dir_name,
            modal_periods=modal_periods,
            modal_summary=modal_summary,
            metrics=metrics,
            combined_center_drifts=combined_center_drifts,
        )

    def extract_global_modal_data(self) -> tuple[list[float], list[float], ModalSummary]:
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.0)
        ops.analysis("Static")

        num_modes_required = min(self.context.config.num_modes, self.context.num_stories * 2)
        self.context.logger.info(f"\n[校核器] 正在提取 {num_modes_required} 阶特征值 (UmfPack)...")
        start = time()
        eigen_values = ops.eigen("-genBandArpack", num_modes_required)
        end = time()
        self.context.logger.info(f"Eigenvalue extraction completed in {end - start:.2f} seconds.")

        if isinstance(eigen_values, (int, float)):
            eigen_values = [float(eigen_values)]
        else:
            eigen_values = [float(value) for value in eigen_values if float(value) > 1.0e-12]

        returned = ops.modalProperties("-return")
        if not isinstance(returned, dict):
            raise RuntimeError("modalProperties('-return') 未返回 dict，无法基于质量参与系数识别主导模态")

        modal_periods = modal_periods_from_eigenvalues(eigen_values)
        self.context.logger.info(f"Modal periods (s): {[round(period, 4) for period in modal_periods]}")
        return eigen_values, modal_periods, self.extract_modal_summary(modal_periods, returned)
