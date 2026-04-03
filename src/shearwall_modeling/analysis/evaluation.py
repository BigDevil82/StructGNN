import logging

from ..builders.base import ModelBuildResult
from ..core.config import ModelConfig
from .checkers import (
    DirectionChecker,
    InterstoryDriftChecker,
    PeriodRatioChecker,
    ShearWeightRatioChecker,
    StiffnessChecker,
    TorsionChecker,
)
from .response_spectrum import AnalysisModelContext, ResponseSpectrumAnalyzer
from .results import DirectionCheckResult, DirectionResponse, WallAxialMetric, init_direction_check_result
from .wall_axial import WallAxialCompressionChecker


class EvaluationReportPrinter:
    def __init__(self, logger: logging.Logger, min_shear_ratio: float):
        self.logger = logger
        self.min_shear_ratio = min_shear_ratio

    def print(
        self, results: dict[str, DirectionCheckResult], wall_axial_metrics: list[WallAxialMetric]
    ) -> None:
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

        for dir_name, result in results.items():
            self.logger.info(f"\n【{dir_name}向校核结果】")
            self.logger.info(
                f" -> 扭转不规则 (限值 1.2/1.5): {'✅通过' if result.is_torsion_passed else '❌超限'}"
            )
            self.logger.info(
                f" -> 最小剪重比 (限值 {self.min_shear_ratio}): "
                f"{'✅通过' if result.is_shear_weight_passed else '❌超限'}"
            )
            self.logger.info(
                f" -> 刚度突变 (限值 0.7/0.8): {'✅通过' if result.is_stiffness_passed else '❌超限'}"
            )
            self.logger.info(
                " -> 最大层间位移角 "
                f"(限值 1/{int(round(1.0 / result.interstory_drift_limit))}): "
                f"{result.max_interstory_drift_ratio:.6f} "
                f"(控制层: {result.max_interstory_drift_story}层) "
                f"{'✅通过' if result.is_interstory_drift_passed else '❌超限'}"
            )
            self.logger.info("楼层 | 层间位移角Max | 位移比(Max/Avg) | 剪重比(%) | 刚度比γ1 | 刚度比γ2")
            self.logger.info("-" * 78)
            for metric in reversed(result.metrics):
                self.logger.info(
                    f"  {metric.story:02d}  |    {metric.drift_max:.6f}   |     {metric.torsion_ratio:.3f}     "
                    f"|   {metric.shear_weight_ratio*100:.2f}   |  {metric.gamma1:.2f}   |  {metric.gamma2:.2f}"
                )

        if wall_axial_metrics:
            self.logger.info("\n【墙肢轴压比校核】")
            worst = max(wall_axial_metrics, key=lambda item: item.axial_ratio)
            self.logger.info(
                f" 控制墙肢: #{worst.wall_id}, 轴压比={worst.axial_ratio:.3f}, "
                f"限值={worst.ratio_limit:.3f}, {'✅通过' if worst.is_passed else '❌超限'}"
            )
            self.logger.info("墙ID | 轴力(kN) | 面积(m2) | 轴应力(MPa) | 轴压比")
            self.logger.info("-" * 64)
            for item in wall_axial_metrics:
                self.logger.info(
                    f" {item.wall_id:03d} | {item.axial_force_n/1e3:8.2f} | {item.area_m2:7.3f} | "
                    f" {item.axial_stress_mpa:9.3f} | {item.axial_ratio:6.3f}"
                )


class SeismicEvaluationPipeline:
    """Coordinate modal response extraction and code checks."""

    def __init__(self, build_result: ModelBuildResult, config: ModelConfig, logger: logging.Logger):
        self.context = AnalysisModelContext(build_result=build_result, config=config, logger=logger)
        self.analyzer = ResponseSpectrumAnalyzer(self.context)
        shear_weight_checker = ShearWeightRatioChecker.from_intensity(config.seismic.intensity)
        self.direction_checkers: list[DirectionChecker] = [
            TorsionChecker(),
            shear_weight_checker,
            StiffnessChecker(),
            PeriodRatioChecker(),
            InterstoryDriftChecker(),
        ]
        self.wall_axial_checker = WallAxialCompressionChecker(self.context)
        self.report_printer = EvaluationReportPrinter(logger, shear_weight_checker.min_ratio)

    def _evaluate_direction(self, response: DirectionResponse) -> DirectionCheckResult:
        result = init_direction_check_result(response)
        for checker in self.direction_checkers:
            checker.apply(response, result)
        return result

    def evaluate(
        self,
    ) -> tuple[dict[str, list[float]], dict[str, DirectionCheckResult], list[WallAxialMetric]]:
        eigen_values, modal_periods, modal_summary = self.analyzer.extract_global_modal_data()
        story_weights = self.analyzer.compute_story_weights()

        check_results: dict[str, DirectionCheckResult] = {}
        for dir_idx, dir_name in ((1, "X"), (2, "Y")):
            response = self.analyzer.analyze_direction(
                dir_idx=dir_idx,
                dir_name=dir_name,
                eigen_values=eigen_values,
                modal_periods=modal_periods,
                modal_summary=modal_summary,
                story_weights=story_weights,
            )
            check_results[dir_name] = self._evaluate_direction(response)

        wall_axial_metrics = self.wall_axial_checker.check()
        return check_results, wall_axial_metrics


class SeismicCodeChecker:
    """Backward-compatible facade over the evaluation pipeline."""

    def __init__(self, build_result: ModelBuildResult, config: ModelConfig, logger: logging.Logger):
        self.pipeline = SeismicEvaluationPipeline(build_result, config, logger)
        self.wall_axial_metrics: list[WallAxialMetric] = []

    def run_analysis_and_evaluate(self) -> tuple[dict[str, list[float]], dict[str, DirectionCheckResult]]:
        check_results, wall_axial_metrics = self.pipeline.evaluate()
        self.wall_axial_metrics = wall_axial_metrics
        return check_results

    def _print_report(self, results: dict[str, DirectionCheckResult]) -> None:
        self.pipeline.report_printer.print(results, self.wall_axial_metrics)
