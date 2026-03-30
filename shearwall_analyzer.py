import logging
import os
from pathlib import Path
from typing import Optional

import openseespy.opensees as ops

from misc.logger import setup_file_logger
from misc.parallel import run_batch
from misc.timer import Timer
from shearwall_modeling import ModelConfig, SectionConfig, StandardStoryGroupConfig, load_and_scale_input
from shearwall_modeling.builders import DetailedShellBuilder
from shearwall_modeling.evaluation import SeismicCodeChecker

logger = setup_file_logger("ShearwallAnalyzer", Path("result/logs/shearwall_analyzer.log"), logging.INFO)
logger.info("\n\n\n\nSHEARWALL ANALYZER STARTED\n")


def build_single(json_path: Path) -> None:
    # User-configurable inputs for parameterized invocation.
    # json_path = Path(r"dxf\cad_json_data\shearwall_split_8_2\L1L28_39.json")
    # json_path = Path(r"result\case_study\L27_136_fem_data.json")
    input_unit_scale_to_m = 0.001  # JSON coordinates are in mm.
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

    standard_story_groups = [
        StandardStoryGroupConfig(count=10, story_height=3.0),
        # StandardStoryGroupConfig(
        #     count=6,
        #     story_height=3.3,
        #     section=SectionConfig(wall_thickness=0.25, beam_width=0.35, beam_depth=0.6, slab_thickness=0.12),
        # ),
        # StandardStoryGroupConfig(
        #     count=6,
        #     story_height=3.6,
        #     section=SectionConfig(wall_thickness=0.30, beam_width=0.40, beam_depth=0.7, slab_thickness=0.14),
        # ),
    ]

    config = ModelConfig(
        num_modes=num_modes,
        standard_story_groups=standard_story_groups,
    )
    config.seismic.combination_method = combine_method

    builder = DetailedShellBuilder(logger)
    build_result = builder.build(input_data, config)

    # 2. 挂载综合校核器 (它将接管特征值求解和反应谱迭代)
    checker = SeismicCodeChecker(build_result.master_nodes, config, logger)

    # 一次性返回：普通的质心层间位移角(兼容原有需求) + 规范校核结果
    drifts, checks = checker.run_analysis_and_evaluate()

    # 打印层间位移角(‰)
    # print(f"\n{config.seismic.combination_method} inter-story drift ratios (‰):")
    # for direction in ("X", "Y"):
    #     print(f"  {direction}: {[round(v * 1000.0, 2) for v in drifts[direction]]}")

    ops.wipe()


def _analyze_task(json_path: str) -> Optional[str]:
    """并行工作函数：成功返回 None，失败返回错误信息。"""
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
        for f in failed:
            logger.info(f"{f}")


if __name__ == "__main__":
    # build_single(Path(r"dxf\cad_json_data\shearwall_split_8_2\L27_59.json"))
    build_folder(r"dxf\cad_json_data\fem_raw")
