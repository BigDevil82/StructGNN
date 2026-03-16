from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline_from_json


def pyyjks():
    """YJK 脚本入口：读取 JSON 完成建模、分析和结果提取。"""
    config = PipelineConfig()

    json_path = Path(__file__).resolve().parents[2] / "result" / "case_study" / "archi_comp_fem_data.json"

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
