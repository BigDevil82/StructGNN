from __future__ import annotations

import importlib
import os
import sys
from datetime import datetime
from pathlib import Path


def _to_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_runtime_symbols(force_reload: bool = False):
    """按当前运行模式动态加载模块，必要时强制重载以避免 yjks_pyload 模块缓存。"""
    try:
        from . import config as config_mod
        from . import pipeline as pipeline_mod
    except ImportError:
        import config as config_mod
        import pipeline as pipeline_mod

    if force_reload:
        config_mod = importlib.reload(config_mod)
        pipeline_mod = importlib.reload(pipeline_mod)

    return config_mod.PipelineConfig, pipeline_mod.run_pipeline_from_json, config_mod, pipeline_mod


def _append_lines(file_path: Path, lines: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def pyyjks():
    """YJK 脚本入口：读取 JSON 完成建模、分析和结果提取。"""
    force_reload = _to_bool_env("YJK_PIPELINE_FORCE_RELOAD", default=True)
    enable_diag = _to_bool_env("YJK_PIPELINE_DIAG", default=True)

    PipelineConfig, run_pipeline_from_json, config_mod, pipeline_mod = _load_runtime_symbols(
        force_reload=force_reload
    )

    config = PipelineConfig()

    # 仅在显式传入环境变量时覆盖配置文件中的 output_dir，避免掩盖 config.py 改动
    output_override = os.getenv("YJK_PIPELINE_OUTPUT")
    if output_override:
        config.output.output_dir = Path(output_override)

    json_path = Path(os.getenv("YJK_PIPELINE_JSON", str(config.model.json_path))).resolve()

    diag_file = Path(
        r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\result\case_study\debug\yjk_pipeline_diag.log"
    ).resolve()

    diag_lines = [
        "=" * 80,
        f"[YJK_PIPELINE][DIAG] ts={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"[YJK_PIPELINE][DIAG] cwd={Path.cwd()}",
        f"[YJK_PIPELINE][DIAG] entry_file={Path(__file__).resolve()}",
        f"[YJK_PIPELINE][DIAG] config_module={getattr(config_mod, '__file__', '<unknown>')}",
        f"[YJK_PIPELINE][DIAG] pipeline_module={getattr(pipeline_mod, '__file__', '<unknown>')}",
        f"[YJK_PIPELINE][DIAG] force_reload={force_reload}",
        f"[YJK_PIPELINE][DIAG] model.story_num={config.model.story_num}",
        f"[YJK_PIPELINE][DIAG] model.story_height={config.model.story_height}",
        f"[YJK_PIPELINE][DIAG] output.output_dir(raw)={config.output.output_dir}",
        f"[YJK_PIPELINE][DIAG] json_path={json_path}",
        f"[YJK_PIPELINE][DIAG] diag_file={diag_file}",
    ]

    if enable_diag:
        if "config" in sys.modules:
            diag_lines.append(
                "[YJK_PIPELINE][DIAG] sys.modules['config']="
                f"{getattr(sys.modules['config'], '__file__', '<unknown>')}"
            )
        if "yjk_pipeline.config" in sys.modules:
            diag_lines.append(
                "[YJK_PIPELINE][DIAG] sys.modules['yjk_pipeline.config']="
                f"{getattr(sys.modules['yjk_pipeline.config'], '__file__', '<unknown>')}"
            )
        _append_lines(diag_file, diag_lines)

    result = run_pipeline_from_json(json_path, config)

    summary_lines = [
        (
            "[YJK_PIPELINE][RESULT] "
            f"build={result.build_success}, "
            f"analysis={result.analysis_success}, "
            f"extraction={result.extraction_success}"
        )
    ]

    if result.output_files:
        for key, value in result.output_files.items():
            summary_lines.append(f"[YJK_PIPELINE][RESULT] {key}={value}")

    _append_lines(diag_file, summary_lines)

    return result


if __name__ == "__main__":
    pyyjks()
