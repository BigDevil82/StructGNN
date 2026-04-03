import logging
import math
from dataclasses import dataclass
from time import time
from typing import Any

import openseespy.opensees as ops

from .builders.base import ModelBuildResult
from .config import ModelConfig
from .constants import CONCRETE_COMPRESSIVE_STRENGTH_PA, MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY
from .modal_combination import cqc, srss


@dataclass
class StoryMetric:
    story: int
    drift_max: float
    drift_avg: float
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


class SeismicCodeChecker:
    """
    独立于标准分析的综合指标校核器 (高级分析引擎模式)。
    仅需执行一次循环，同时输出常规层间位移角和规范抗震指标。
    """

    def __init__(self, build_result: ModelBuildResult, config: ModelConfig, logger: logging.Logger):
        self.master_nodes = build_result.master_nodes
        self.floor_area = build_result.floor_area
        self.wall_base_units = build_result.wall_base_units
        self.floor_story_nodes = build_result.floor_story_nodes
        self.config = config
        self.num_stories = len(self.master_nodes)
        self.story_heights = config.get_story_heights()
        self.story_profiles = config.resolve_story_profiles()
        self.logger = logger

        self.xmin, self.xmax, self.ymin, self.ymax = self._get_model_bbox()
        self.floor_masses = [ops.nodeMass(n, 1) for n in self.master_nodes]
        self.min_shear_ratio = self._get_min_shear_ratio(config.seismic.intensity)
        self.max_interstory_drift_limit = 1.0 / 1000.0
        self.wall_axial_metrics: list[WallAxialMetric] = []

    def _concrete_fc_pa(self, concrete_grade: str) -> float:
        grade = concrete_grade.strip().upper()
        if grade not in CONCRETE_COMPRESSIVE_STRENGTH_PA:
            raise ValueError(f"Unsupported concrete grade for axial check: {concrete_grade}")
        return CONCRETE_COMPRESSIVE_STRENGTH_PA[grade]

    def _run_wall_axial_check(self) -> list[WallAxialMetric]:
        ts_tag = 70001
        pat_tag = 70001

        floor_gravity_forces = [
            (p.mass_source.dead_kpa + 0.5 * p.mass_source.live_kpa) * 1000.0 * self.floor_area
            for p in self.story_profiles
        ]

        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)
        for i, floor_nodes in enumerate(self.floor_story_nodes):
            nodal_force = floor_gravity_forces[i] / len(floor_nodes)
            for node in floor_nodes:
                ops.load(node, 0.0, 0.0, -nodal_force, 0.0, 0.0, 0.0)

        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")

        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(f"Static gravity case failed with code={ok}")

        ops.reactions()
        base_nodes = sorted({n for unit in self.wall_base_units for n in unit.base_nodes})
        node_rxn = {n: ops.nodeReaction(n, 3) for n in base_nodes}

        node_share = {n: 0 for n in base_nodes}
        for unit in self.wall_base_units:
            for n in unit.base_nodes:
                node_share[n] += 1

        fc_pa = self._concrete_fc_pa(self.story_profiles[0].material.concrete_grade)
        ratio_limit = self.config.seismic.axial_compression_ratio_limit
        metrics: list[WallAxialMetric] = []

        for unit in self.wall_base_units:
            axial_force_n = sum(node_rxn[n] / node_share[n] for n in unit.base_nodes)
            area_m2 = unit.length * unit.thickness
            stress_pa = axial_force_n / area_m2
            ratio = stress_pa / fc_pa
            metrics.append(
                WallAxialMetric(
                    wall_id=unit.wall_id,
                    axial_force_n=axial_force_n,
                    area_m2=area_m2,
                    axial_stress_mpa=stress_pa / 1.0e6,
                    axial_ratio=ratio,
                    ratio_limit=ratio_limit,
                    is_passed=ratio <= ratio_limit,
                )
            )

        return metrics

    def _get_model_bbox(self) -> tuple[float, float, float, float]:
        nodes = ops.getNodeTags()
        xs = [ops.nodeCoord(n, 1) for n in nodes]
        ys = [ops.nodeCoord(n, 2) for n in nodes]
        return min(xs), max(xs), min(ys), max(ys)

    def _get_min_shear_ratio(self, intensity: float) -> float:
        key = round(float(intensity), 2)
        return MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY.get(key, 0.016)

    def _combine(self, modal_vals: list[float], eigs: list[float]) -> float:
        num_modes = len(modal_vals)
        damping = [self.config.seismic.damping_ratio] * num_modes
        scale_factors = [1.0] * num_modes
        if self.config.seismic.combination_method.upper() == "CQC":
            return cqc(modal_vals, eigs, damping, scale_factors)
        return srss(modal_vals, scale_factors)

    def _extract_modal_summary(
        self, periods: list[float], modal_props: dict[str, Any]
    ) -> tuple[int | None, float | None, int | None, float | None, float | None, bool]:
        translational_mode_index = None
        translational_period = None
        torsional_mode_index = None
        torsional_period = None

        self.logger.info("\n[校核器] 正在根据模态参与质量比识别主导模态...")
        translational_mode_index, torsional_mode_index = self._identify_modes_by_participation(modal_props)
        if translational_mode_index is not None and translational_mode_index - 1 < len(periods):
            translational_period = periods[translational_mode_index - 1]
        if torsional_mode_index is not None and torsional_mode_index - 1 < len(periods):
            torsional_period = periods[torsional_mode_index - 1]

        period_ratio = None
        is_period_ratio_passed = False
        if translational_period and torsional_period:
            period_ratio = torsional_period / translational_period
            is_period_ratio_passed = period_ratio < 0.9

        return (
            translational_mode_index,
            translational_period,
            torsional_mode_index,
            torsional_period,
            period_ratio,
            is_period_ratio_passed,
        )

    def _safe_float_list(self, values: Any, max_len: int) -> list[float]:
        if not isinstance(values, list):
            return [0.0] * max_len
        out: list[float] = []
        for value in values[:max_len]:
            try:
                out.append(abs(float(value)))
            except (TypeError, ValueError):
                out.append(0.0)
        if len(out) < max_len:
            out.extend([0.0] * (max_len - len(out)))
        return out

    def _identify_modes_by_participation(self, modal_props: dict[str, Any]) -> tuple[int | None, int | None]:
        num_modes = len(modal_props.get("eigenLambda", []))
        if num_modes <= 0:
            return None, None

        mx = self._safe_float_list(modal_props.get("partiMassRatiosMX"), num_modes)
        my = self._safe_float_list(modal_props.get("partiMassRatiosMY"), num_modes)
        rmz = self._safe_float_list(modal_props.get("partiMassRatiosRMZ"), num_modes)

        translational_mode_index = None
        torsional_mode_index = None

        for i in range(num_modes):
            trans_ratio = max(mx[i], my[i])
            tors_ratio = rmz[i]
            if translational_mode_index is None and trans_ratio >= tors_ratio and trans_ratio > 0.0:
                translational_mode_index = i + 1
            if torsional_mode_index is None and tors_ratio > trans_ratio and tors_ratio > 0.0:
                torsional_mode_index = i + 1
            if translational_mode_index is not None and torsional_mode_index is not None:
                break

        return translational_mode_index, torsional_mode_index

    def _compute_story_weights(self) -> list[float]:
        story_weights = [0.0] * self.num_stories
        cumulative_mass = 0.0
        for story_index in reversed(range(self.num_stories)):
            cumulative_mass += self.floor_masses[story_index]
            story_weights[story_index] = cumulative_mass * self.config.seismic.gravity
        return story_weights

    def _compute_corner_displacements(
        self, dir_idx: int, cm_disp: float, rotation_z: float, cm_x: float, cm_y: float
    ) -> list[float]:
        dx = [self.xmin - cm_x, self.xmax - cm_x, self.xmax - cm_x, self.xmin - cm_x]
        dy = [self.ymin - cm_y, self.ymin - cm_y, self.ymax - cm_y, self.ymax - cm_y]
        if dir_idx == 1:
            return [cm_disp - rotation_z * delta_y for delta_y in dy]
        return [cm_disp + rotation_z * delta_x for delta_x in dx]

    def _analyze_response_spectrum_mode(
        self,
        dir_idx: int,
        eigen_value: float,
        mode_index: int,
        modal_drift_cm: list[list[float]],
        modal_drift_corners: list[list[list[float]]],
        modal_story_shear: list[list[float]],
    ) -> None:
        ops.responseSpectrumAnalysis(
            dir_idx,
            "-Tn",
            *self.config.seismic.periods,
            "-Sa",
            *self.config.seismic.spectral_accel,
            "-mode",
            mode_index,
        )

        prev_u_cm = 0.0
        prev_u_corners = [0.0] * 4
        inertial_forces = [0.0] * self.num_stories
        for story_index, master in enumerate(self.master_nodes):
            u_cm = ops.nodeDisp(master, dir_idx)
            rotation_z = ops.nodeDisp(master, 6)
            cm_x, cm_y = ops.nodeCoord(master, 1), ops.nodeCoord(master, 2)

            inertial_forces[story_index] = self.floor_masses[story_index] * eigen_value * u_cm
            corner_displacements = self._compute_corner_displacements(dir_idx, u_cm, rotation_z, cm_x, cm_y)
            story_height = self.story_heights[story_index]

            modal_drift_cm[story_index].append((u_cm - prev_u_cm) / story_height)
            for corner_index, corner_disp in enumerate(corner_displacements):
                modal_drift_corners[corner_index][story_index].append(
                    (corner_disp - prev_u_corners[corner_index]) / story_height
                )

            prev_u_cm = u_cm
            prev_u_corners = corner_displacements

        cumulative_shear = 0.0
        for story_index in reversed(range(self.num_stories)):
            cumulative_shear += inertial_forces[story_index]
            modal_story_shear[story_index].append(cumulative_shear)

    def _update_story_stiffness_ratios(self, metrics: list[StoryMetric], stiffness_values: list[float]) -> None:
        for story_index, metric in enumerate(metrics):
            current_stiffness = stiffness_values[story_index]
            next_stiffness = (
                stiffness_values[story_index + 1] if story_index < self.num_stories - 1 else current_stiffness
            )

            upper_stiffness = [
                stiffness_values[story_index + offset]
                for offset in range(1, 4)
                if story_index + offset < self.num_stories
            ]
            average_upper_stiffness = (
                sum(upper_stiffness) / len(upper_stiffness) if upper_stiffness else current_stiffness
            )
            metric.stiffness_ratio_adjacent = (
                current_stiffness / next_stiffness if next_stiffness > 1.0e-9 else 1.0
            )
            metric.stiffness_ratio_average = (
                current_stiffness / average_upper_stiffness if average_upper_stiffness > 1.0e-9 else 1.0
            )

    def _evaluate_direction(
        self,
        dir_idx: int,
        dir_name: str,
        eigs: list[float],
        periods: list[float],
        modal_summary: tuple[int | None, float | None, int | None, float | None, float | None, bool],
        story_weights: list[float],
    ) -> tuple[list[float], DirectionCheckResult]:
        modal_drift_cm = [[] for _ in range(self.num_stories)]
        modal_drift_corners = [[[] for _ in range(self.num_stories)] for _ in range(4)]
        modal_story_shear = [[] for _ in range(self.num_stories)]

        for mode_index, eigen_value in enumerate(eigs, start=1):
            self._analyze_response_spectrum_mode(
                dir_idx=dir_idx,
                eigen_value=eigen_value,
                mode_index=mode_index,
                modal_drift_cm=modal_drift_cm,
                modal_drift_corners=modal_drift_corners,
                modal_story_shear=modal_story_shear,
            )

        metrics: list[StoryMetric] = []
        stiffness_values = [0.0] * self.num_stories
        combined_cm_drifts: list[float] = []
        for story_index in range(self.num_stories):
            story_shear = self._combine(modal_story_shear[story_index], eigs)
            cm_drift = abs(self._combine(modal_drift_cm[story_index], eigs))
            combined_cm_drifts.append(cm_drift)

            corner_drifts = [
                abs(self._combine(modal_drift_corners[corner_index][story_index], eigs))
                for corner_index in range(4)
            ]
            drift_max = max(corner_drifts)
            drift_min = min(corner_drifts)
            drift_avg = 0.5 * (drift_max + drift_min)
            torsion_ratio = drift_max / drift_avg if drift_avg > 1.0e-9 else 1.0
            shear_weight_ratio = story_shear / story_weights[story_index] if story_weights[story_index] > 1.0e-9 else 0.0
            stiffness = story_shear / cm_drift if cm_drift > 1.0e-9 else 0.0
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

        self._update_story_stiffness_ratios(metrics, stiffness_values)
        max_drift_metric = max(metrics, key=lambda metric: metric.drift_max)
        (
            translational_mode_index,
            translational_period,
            torsional_mode_index,
            torsional_period,
            period_ratio,
            is_period_ratio_passed,
        ) = modal_summary
        result = DirectionCheckResult(
            direction=dir_name,
            metrics=metrics,
            modal_periods=periods.copy(),
            translational_mode_index=translational_mode_index,
            translational_period=translational_period,
            torsional_mode_index=torsional_mode_index,
            torsional_period=torsional_period,
            period_ratio=period_ratio,
            is_torsion_passed=all(metric.torsion_ratio <= 1.5 for metric in metrics),
            is_shear_weight_passed=all(
                metric.shear_weight_ratio >= self.min_shear_ratio for metric in metrics
            ),
            is_stiffness_passed=all(
                metric.stiffness_ratio_adjacent >= 0.7 and metric.stiffness_ratio_average >= 0.8
                for metric in metrics
            ),
            is_period_ratio_passed=is_period_ratio_passed,
            max_interstory_drift_ratio=max_drift_metric.drift_max,
            max_interstory_drift_story=max_drift_metric.story,
            interstory_drift_limit=self.max_interstory_drift_limit,
            is_interstory_drift_passed=max_drift_metric.drift_max <= self.max_interstory_drift_limit,
        )
        return combined_cm_drifts, result

    def run_analysis_and_evaluate(self) -> tuple[dict[str, list[float]], dict[str, DirectionCheckResult]]:
        """执行分析并同时提取所有需要的数据，绝不重复计算"""
        # 1. 设置求解器(引入UmfPack提速)
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.0)
        ops.analysis("Static")

        num_modes_required = min(self.config.num_modes, self.num_stories * 2)
        self.logger.info(f"\n[校核器] 正在提取 {num_modes_required} 阶特征值 (UmfPack)...")
        start = time()
        eigs = ops.eigen("-genBandArpack", num_modes_required)
        end = time()
        self.logger.info(f"Eigenvalue extraction completed in {end - start:.2f} seconds.")

        if isinstance(eigs, (int, float)):
            eigs = [float(eigs)]
        else:
            eigs = [float(x) for x in eigs if float(x) > 1e-12]

        returned = ops.modalProperties("-return")
        if not isinstance(returned, dict):
            raise RuntimeError("modalProperties('-return') 未返回 dict，无法基于质量参与系数识别主导模态")
        modal_props: dict[str, Any] = returned

        periods = [2.0 * math.pi / math.sqrt(lam) for lam in eigs]
        self.logger.info(f"Modal periods (s): {[round(t, 4) for t in periods]}")
        modal_summary = self._extract_modal_summary(periods, modal_props)

        story_weights = self._compute_story_weights()

        standard_drifts = {}
        check_results = {}

        for dir_idx, dir_name in ((1, "X"), (2, "Y")):
            standard_drifts[dir_name], check_results[dir_name] = self._evaluate_direction(
                dir_idx=dir_idx,
                dir_name=dir_name,
                eigs=eigs,
                periods=periods,
                modal_summary=modal_summary,
                story_weights=story_weights,
            )

        self.wall_axial_metrics = self._run_wall_axial_check()

        # self._print_report(check_results)
        return standard_drifts, check_results

    def _print_report(self, results: dict[str, DirectionCheckResult]):
        self.logger.info("\n" + "=" * 50)
        self.logger.info("结构抗震规范核心指标综合校核报告")
        self.logger.info("=" * 50)

        first_result = next(iter(results.values()), None)
        if first_result is not None:
            self.logger.info("\n【模态结果】")
            self.logger.info(" 前n阶周期:")
            for idx, period in enumerate(first_result.modal_periods, start=1):
                self.logger.info(f"  第{idx}阶: {period:.4f} s")

            if first_result.period_ratio is not None:
                self.logger.info(
                    " 首个平动周期/首个扭转周期: "
                    f"第{first_result.translational_mode_index}阶 {first_result.translational_period:.4f} s / "
                    f"第{first_result.torsional_mode_index}阶 {first_result.torsional_period:.4f} s"
                )
                self.logger.info(
                    " 周期比 T_torsion / T_translation < 0.9: "
                    f"{first_result.period_ratio:.3f} "
                    f"({'✅通过' if first_result.is_period_ratio_passed else '❌超限'})"
                )
            else:
                self.logger.info(" 首个平动或扭转主导模态未识别，周期比无法校核。")

        for dir_name, res in results.items():
            self.logger.info(f"\n【{dir_name}向校核结果】")
            self.logger.info(
                f" -> 扭转不规则 (限值 1.2/1.5): {'✅通过' if res.is_torsion_passed else '❌超限'}"
            )
            self.logger.info(
                f" -> 最小剪重比 (限值 {self.min_shear_ratio}): {'✅通过' if res.is_shear_weight_passed else '❌超限'}"
            )
            self.logger.info(
                f" -> 刚度突变 (限值 0.7/0.8): {'✅通过' if res.is_stiffness_passed else '❌超限'}"
            )
            self.logger.info(
                " -> 最大层间位移角 "
                f"(限值 1/{int(round(1.0 / res.interstory_drift_limit))}): "
                f"{res.max_interstory_drift_ratio:.6f} "
                f"(控制层: {res.max_interstory_drift_story}层) "
                f"{'✅通过' if res.is_interstory_drift_passed else '❌超限'}"
            )

            self.logger.info("楼层 | 层间位移角Max | 位移比(Max/Avg) | 剪重比(%) | 刚度比γ1 | 刚度比γ2")
            self.logger.info("-" * 78)
            # 逆序输出，符合结构从上到下的阅读直觉
            for m in reversed(res.metrics):
                self.logger.info(
                    f"  {m.story:02d}  |    {m.drift_max:.6f}   |     {m.torsion_ratio:.3f}     |   {m.shear_weight_ratio*100:.2f}   |  {m.gamma1:.2f}   |  {m.gamma2:.2f}"
                )

        if self.wall_axial_metrics:
            self.logger.info("\n【墙肢轴压比校核】")
            worst = max(self.wall_axial_metrics, key=lambda item: item.axial_ratio)
            self.logger.info(
                f" 控制墙肢: #{worst.wall_id}, 轴压比={worst.axial_ratio:.3f}, "
                f"限值={worst.ratio_limit:.3f}, {'✅通过' if worst.is_passed else '❌超限'}"
            )
            self.logger.info("墙ID | 轴力(kN) | 面积(m2) | 轴应力(MPa) | 轴压比")
            self.logger.info("-" * 64)
            for item in self.wall_axial_metrics:
                self.logger.info(
                    f" {item.wall_id:03d} | {item.axial_force_n/1e3:8.2f} | {item.area_m2:7.3f} | "
                    f" {item.axial_stress_mpa:9.3f} | {item.axial_ratio:6.3f}"
                )
