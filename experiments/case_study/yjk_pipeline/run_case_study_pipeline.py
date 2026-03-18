from __future__ import annotations

import os
from pathlib import Path

try:
    from .config import PipelineConfig
    from .pipeline import run_pipeline_from_json
except ImportError:
    from config import PipelineConfig
    from pipeline import run_pipeline_from_json


def pyyjks():
    """YJK 脚本入口：读取 JSON 完成建模、分析和结果提取。"""
    config = PipelineConfig()

    project_root = Path(__file__).resolve().parents[3]
    default_output_dir = project_root / "result" / "case_study" / "yjk_pipeline_output"
    output_dir = Path(os.getenv("YJK_PIPELINE_OUTPUT", str(default_output_dir)))
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    config.output.output_dir = output_dir

    default_json_path = (
        project_root / "result" / "case_study" / "building_components_fem_data.json"
    )
    json_path = Path(os.getenv("YJK_PIPELINE_JSON", str(default_json_path))).resolve()

    result = run_pipeline_from_json(json_path, config)

    print(
        "Pipeline done: "
        f"build={result.build_success}, "
        f"analysis={result.analysis_success}, "
        f"extraction={result.extraction_success}"
    )

    if result.output_files:
        for key, value in result.output_files.items():
            print(f"{key}: {value}")

    return result


if __name__ == "__main__":
    pyyjks()
