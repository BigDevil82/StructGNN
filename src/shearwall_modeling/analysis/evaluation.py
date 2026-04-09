import logging

from ..builders.base import AnalysisModelContext, ModelBuildResult
from ..core.config import ModelConfig
from ..core.constants import MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY
from .checkers import (
    DirectionChecker,
    InterstoryDriftChecker,
    PeriodRatioChecker,
    ShearWeightRatioChecker,
    StiffnessChecker,
    TorsionChecker,
)
from .response_spectrum import ResponseSpectrumAnalyzer
from .results import (
    AnalysisResult,
    DirectionCheckResult,
    DirectionResponse,
    GravityCaseResult,
    OverallCheckResult,
    ResponseSpectrumCaseResult,
    init_direction_check_result,
)
from .wall_axial import GravityCaseAnalyzer


class AnalysisResultBuilder:
    def __init__(self, context: AnalysisModelContext):
        self.context = context
        self.rsa_analyzer = ResponseSpectrumAnalyzer(context)
        self.gravity_analyzer = GravityCaseAnalyzer(context)

    def build(self) -> AnalysisResult:
        rsa_result = self.rsa_analyzer.run()
        gravity_result = self.gravity_analyzer.run()
        return self._compose_snapshot(rsa_result, gravity_result)

    def _compose_snapshot(
        self, rsa_result: ResponseSpectrumCaseResult, gravity_result: GravityCaseResult
    ) -> AnalysisResult:
        return AnalysisResult(
            eigen_values=rsa_result.eigen_values,
            modal_periods=rsa_result.modal_periods,
            modal_summary=rsa_result.modal_summary,
            story_weights=rsa_result.story_weights,
            direction_responses=rsa_result.direction_responses,
            gravity_beam_forces=gravity_result.gravity_beam_forces,
            gravity_wall_forces=gravity_result.gravity_wall_forces,
            seismic_beam_forces=rsa_result.seismic_beam_forces,
            seismic_wall_forces=rsa_result.seismic_wall_forces,
            wall_axial_metrics=gravity_result.wall_axial_metrics,
        )


class SeismicEvaluationPipeline:
    """Coordinate modal response extraction and code checks."""

    def __init__(self, config: ModelConfig):
        shear_weight_checker = ShearWeightRatioChecker.from_intensity(config.seismic.intensity)
        self.direction_checkers: list[DirectionChecker] = [
            TorsionChecker(),
            shear_weight_checker,
            StiffnessChecker(),
            InterstoryDriftChecker(),
        ]
        self.period_ratio_checker = PeriodRatioChecker()

    def _evaluate_direction(self, response: DirectionResponse) -> DirectionCheckResult:
        result = init_direction_check_result(response)
        for checker in self.direction_checkers:
            checker.apply(response, result)

        return result

    def evaluate(self, anysis_result: AnalysisResult) -> tuple[OverallCheckResult, dict[str, DirectionCheckResult]]:

        overall_result = OverallCheckResult()
        dir_chk_results: dict[str, DirectionCheckResult] = {}
        for dir_name in ("X", "Y"):
            response = anysis_result.direction_responses[dir_name]
            dir_chk_results[dir_name] = self._evaluate_direction(response)

        # aggregate directional results to overall result
        overall_result.is_torsion_passed = all(result.is_torsion_passed for result in dir_chk_results.values())
        overall_result.is_shear_weight_passed = all(
            result.is_shear_weight_passed for result in dir_chk_results.values()
        )
        overall_result.is_stiffness_passed = all(result.is_stiffness_passed for result in dir_chk_results.values())
        overall_result.is_interstory_drift_passed = all(
            result.is_interstory_drift_passed for result in dir_chk_results.values()
        )

        self.period_ratio_checker.apply(anysis_result.modal_summary, overall_result)
        return overall_result, dir_chk_results


class EvaluationReportPrinter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def print(
        self,
        config: ModelConfig,
        analysis_result: AnalysisResult,
        overall_result: OverallCheckResult,
        dir_results: dict[str, DirectionCheckResult],
    ) -> None:
        self.logger.info("\n" + "=" * 50)
        self.logger.info("结构抗震规范核心指标综合校核报告")
        self.logger.info("=" * 50)

        modal_summary = analysis_result.modal_summary
        wall_axial_metrics = analysis_result.wall_axial_metrics
        self.min_shear_ratio = MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY[config.seismic.intensity]

        if modal_summary is not None:
            self.logger.info("\n【模态结果】")
            self.logger.info(" 前n阶周期:")
            for idx, period in enumerate(modal_summary.periods, start=1):
                self.logger.info(f"  第{idx}阶: {period:.4f} s")

            if modal_summary.period_ratio is not None:
                self.logger.info(
                    " 首个平动周期/首个扭转周期: "
                    f"第{modal_summary.translational_mode_index}阶 {modal_summary.translational_period:.4f} s / "
                    f"第{modal_summary.torsional_mode_index}阶 {modal_summary.torsional_period:.4f} s"
                )
                self.logger.info(
                    " 周期比 T_torsion / T_translation < 0.9: "
                    f"{modal_summary.period_ratio:.3f} "
                    f"({'✅通过' if overall_result.is_period_ratio_passed else '❌超限'})"
                )
            else:
                self.logger.info(" 首个平动或扭转主导模态未识别，周期比无法校核。")

        for dir_name, result in dir_results.items():
            self.logger.info(f"\n【{dir_name}向校核结果】")
            self.logger.info(f" -> 扭转不规则 (限值 1.2/1.5): {'✅通过' if result.is_torsion_passed else '❌超限'}")
            self.logger.info(
                f" -> 最小剪重比 (限值 {self.min_shear_ratio}): "
                f"{'✅通过' if result.is_shear_weight_passed else '❌超限'}"
            )
            self.logger.info(f" -> 刚度突变 (限值 0.7/0.8): {'✅通过' if result.is_stiffness_passed else '❌超限'}")
            self.logger.info(
                " -> 最大层间位移角 "
                f"(限值 1/{int(round(1.0 / result.interstory_drift_limit))}): "
                f"{result.max_interstory_drift_ratio:.6f} "
                f"(控制层: {result.max_interstory_drift_story}层) "
                f"{'✅通过' if result.is_interstory_drift_passed else '❌超限'}"
            )
            self.logger.info("楼层 | 层间位移角Max | 位移比(Max/Avg) | 剪重比(%) | 刚度比γ1 | 刚度比γ2")
            self.logger.info("-" * 60)
            for metric in reversed(result.metrics):
                self.logger.info(
                    f"  {metric.story:02d} | {metric.drift_max:.6f} | {metric.torsion_ratio:.3f} "
                    f"| {metric.shear_weight_ratio*100:.2f} | {metric.gamma1:.2f} | {metric.gamma2:.2f}"
                )

        if wall_axial_metrics:
            self.logger.info("\n【墙肢轴压比校核】")
            worst = max(wall_axial_metrics, key=lambda item: item.axial_ratio)
            self.logger.info(
                f" 控制墙肢: #{worst.wall_id}, 轴压比={worst.axial_ratio:.3f}, "
                f"限值={worst.ratio_limit:.3f}, {'✅通过' if worst.is_passed else '❌超限'}"
            )
            if not worst.is_passed:
                self.logger.info("墙ID | 轴力(kN) | 面积(m2) | 轴应力(MPa) | 轴压比")
                self.logger.info("-" * 64)
                for item in [m for m in wall_axial_metrics if not m.is_passed]:
                    self.logger.info(
                        f" {item.wall_id:03d} | {item.axial_force_n/1e3:8.2f} | {item.area_m2:7.3f} | "
                        f" {item.axial_stress_mpa:9.3f} | {item.axial_ratio:6.3f}"
                    )
