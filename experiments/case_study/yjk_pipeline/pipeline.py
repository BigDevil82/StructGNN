from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .analysis_runner import YJKAnalysisRunner
from .common import clean_output_files, setup_file_logger
from .config import PipelineConfig
from .json_loader import load_segments_from_json, scale_segments, translate_segments_to_origin
from .model_builder import YJKJsonModelBuilder
from .result_extractor import StoryDriftData, YJKResultExtractor


@dataclass
class PipelineResult:
    build_success: bool
    analysis_success: bool
    extraction_success: bool
    drift_data: Optional[StoryDriftData]
    output_files: dict[str, Optional[Path]]


class YJKJsonPipeline:
    """统一流程：读取 JSON -> 建模 -> 分析 -> 结果提取。"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.base_dir = Path(__file__).parent
        self.output_dir = self.config.output.resolve_output_dir(self.base_dir)
        self.logger = setup_file_logger(
            "YJKJsonPipeline",
            self.output_dir / "pipeline.log",
            self.config.output.log_level,
        )

        if self.config.output.clean_old_files:
            clean_output_files(
                self.output_dir,
                [
                    "pipeline.log",
                    "model_builder.log",
                    "analysis_runner.log",
                    "result_extractor.log",
                    "faulthandler.log",
                    "story_drift_report.txt",
                    "story_drift_summary.csv",
                    "key_metrics.csv",
                    "story_drift_plot.png",
                ],
                self.logger,
            )

    def _prepare_segments(self, json_path: Path):
        wall_segments, beam_segments = load_segments_from_json(json_path)

        wall_segments = scale_segments(wall_segments, self.config.model.coord_scale)
        beam_segments = scale_segments(beam_segments, self.config.model.coord_scale)

        translation = (0.0, 0.0)
        if self.config.model.normalize_to_origin:
            wall_segments, beam_segments, translation = translate_segments_to_origin(
                wall_segments,
                beam_segments,
            )

        if not self.config.model.with_beam:
            beam_segments = []

        self.logger.info(
            "Prepared segments from %s. walls=%s, beams=%s, translation=(%.2f, %.2f)",
            json_path,
            len(wall_segments),
            len(beam_segments),
            translation[0],
            translation[1],
        )
        return wall_segments, beam_segments

    def run(self, json_path: Path | str) -> PipelineResult:
        json_path = Path(json_path).resolve()
        if not json_path.exists():
            raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

        self.logger.info("Pipeline started: %s", json_path)

        wall_segments, beam_segments = self._prepare_segments(json_path)

        model_builder = YJKJsonModelBuilder(self.config.model, self.config.output, self.base_dir)
        build_success = model_builder.build_model(wall_segments, beam_segments)

        analysis_success = False
        if build_success and self.config.analysis.run_analysis:
            analysis_runner = YJKAnalysisRunner(self.config.output, self.base_dir)
            analysis_success = analysis_runner.run_full_analysis()
        elif build_success:
            analysis_success = True

        extraction_success = False
        drift_data: Optional[StoryDriftData] = None
        output_files: dict[str, Optional[Path]] = {}

        if build_success and analysis_success:
            with YJKResultExtractor(self.config.extraction, self.config.output, self.base_dir) as extractor:
                drift_data, output_files = extractor.run_extraction()
                extraction_success = True

        self.logger.info(
            "Pipeline finished. build=%s, analysis=%s, extraction=%s",
            build_success,
            analysis_success,
            extraction_success,
        )

        return PipelineResult(
            build_success=build_success,
            analysis_success=analysis_success,
            extraction_success=extraction_success,
            drift_data=drift_data,
            output_files=output_files,
        )


def run_pipeline_from_json(json_path: Path | str, config: Optional[PipelineConfig] = None) -> PipelineResult:
    pipeline = YJKJsonPipeline(config)
    return pipeline.run(json_path)
