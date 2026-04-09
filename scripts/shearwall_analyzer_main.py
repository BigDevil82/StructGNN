import logging
import os
from pathlib import Path
from typing import Optional

import openseespy.opensees as ops

from src.misc.logger import setup_file_logger
from src.misc.parallel import run_batch
from src.misc.timer import Timer
from src.shearwall_modeling import (
    ModelConfig,
    ParametricModelParams,
    StandardStoryGroupConfig,
    build_model_config_from_params,
    load_and_scale_input,
)
from src.shearwall_modeling.analysis.evaluation import (
    AnalysisResultBuilder,
    EvaluationReportPrinter,
    SeismicEvaluationPipeline,
)
from src.shearwall_modeling.builders import DetailedShellBuilder
from src.shearwall_modeling.builders.base import AnalysisModelContext
from src.shearwall_modeling.core.domain import FEMInput
from src.shearwall_modeling.design.pipeline import ReinforcementDesignPipeline

logger = setup_file_logger("ShearwallAnalyzer", Path("outputs/result/logs/shearwall_analyzer.log"), logging.INFO)
logger.info("\n\n\n\nSHEARWALL ANALYZER STARTED\n")


def build_single(json_path: Path) -> None:
    input_unit_scale_to_m = 0.001
    num_modes = 6
    enable_auto_scale = True
    scale_low = 2.0
    scale_high = 6.0
    scale_seed = 42
    manual_scale_factor = None
    combine_method = "CQC"

    input_data, scale = load_and_scale_input(
        json_path=json_path,
        input_unit_scale_to_m=input_unit_scale_to_m,
        enable_auto_scale=enable_auto_scale,
        low=scale_low,
        high=scale_high,
        seed=scale_seed,
        manual_factor=manual_scale_factor,
    )

    logger.info(f"Geometry scale factor used: {scale:.2f}")

    standard_story_groups = [StandardStoryGroupConfig(count=18, story_height=3.0)]
    config = ModelConfig(num_modes=num_modes, standard_story_groups=standard_story_groups)
    config.seismic.combination_method = combine_method

    run_with_config(input_data, config)


def build_single_parametric(json_path: Path, params: ParametricModelParams) -> None:
    input_unit_scale_to_m = 0.001
    enable_auto_scale = True
    scale_low = 2.0
    scale_high = 6.0
    scale_seed = 42
    manual_scale_factor = None
    combine_method = "CQC"

    input_data, scale = load_and_scale_input(
        json_path=json_path,
        input_unit_scale_to_m=input_unit_scale_to_m,
        enable_auto_scale=enable_auto_scale,
        low=scale_low,
        high=scale_high,
        seed=scale_seed,
        manual_factor=manual_scale_factor,
    )

    logger.info(f"Geometry scale factor used: {scale:.2f}")
    config = build_model_config_from_params(params)
    config.seismic.combination_method = combine_method

    run_with_config(input_data, config)


def run_with_config(input_data: FEMInput, config: ModelConfig) -> None:
    builder = DetailedShellBuilder(logger)
    build_result = builder.build(input_data, config)

    context = AnalysisModelContext(build_result=build_result, config=config, logger=logger)
    analysis_result = AnalysisResultBuilder(context).build()

    checker = SeismicEvaluationPipeline(config)
    overall_results, dir_chk_results = checker.evaluate(analysis_result)

    report_printer = EvaluationReportPrinter(logger)
    report_printer.print(config, analysis_result, overall_results, dir_chk_results)

    design_pipeline = ReinforcementDesignPipeline(build_result, analysis_result, config, logger)
    design_summary = design_pipeline.run()

    print("Is passed:", design_summary.is_passed)
    print("Total Concrete Usage(t):", design_summary.total_concrete_kg / 1000)
    print("Total Steel Usage(t):", design_summary.total_steel_kg / 1000)

    print("failed beams:", len(design_summary.failed_beams))
    print("failed walls:", len(design_summary.failed_walls))

    for wall in design_summary.failed_walls:
        wall_id = wall.wall_id
        print(
            f"Failed Wall - ID: {wall.wall_id}, Story: {wall.story}, Messages: {wall.messages}, length: {input_data.walls[wall_id].length:.2f}m"
        )

    ops.wipe()


def _analyze_task(json_path: str) -> Optional[str]:
    try:
        build_single(Path(json_path))
        return None
    except Exception as e:
        return str(e)


def build_folder(dxf_folder: str, max_workers: Optional[int] = None) -> None:
    tasks = []
    for root, _, files in os.walk(dxf_folder):
        for filename in files:
            if filename.lower().endswith(".json"):
                json_path = os.path.join(root, filename)
                tasks.append(json_path)

    if not tasks:
        logger.info(f"No JSON files found in: {dxf_folder}")
        return

    with Timer(prefix="Shearwall batch analysis time:", print_func=logger.info):
        outcomes = run_batch(tasks, _analyze_task, max_workers=max_workers, backend="process")

    failed = []
    for outcome in outcomes:
        json_path = outcome.item
        if outcome.ok and outcome.result is None:
            logger.info(f"Processed: {json_path}")
            continue

        err = outcome.error if outcome.error is not None else (outcome.result or "Unknown error")
        logger.error(f"Error processing {json_path}: {err}")
        failed.append(json_path)

    if failed:
        logger.info("\nFailed to process the following files:")
        for file_path in failed:
            logger.info(f"{file_path}")


def main() -> None:
    # build_folder(r"data\dxf\cad_json_data\fem_raw")
    # build_single_parametric(
    #     Path(r"data\dxf\cad_json_data\fem_raw\L1L28_10.json"),
    #     ParametricModelParams(
    #         N=12,
    #         t_w_bot=250,
    #         h_b=500,
    #         b_b=250,
    #         h_s=120,
    #         conc_bot="C40",
    #         intensity=8.0,
    #         site_class="II",
    #         seismic_group=1,
    #     ),
    # )
    print("Starting single model analysis...")
    with Timer(prefix="Single model analysis time:"):
        build_single(Path(r"data\dxf\cad_json_data\fem_raw_tmp\L1L28_10.json"))
        # build_single(Path(r"outputs\result\case_study\archi_comp_fem_data.json"))


if __name__ == "__main__":
    main()
