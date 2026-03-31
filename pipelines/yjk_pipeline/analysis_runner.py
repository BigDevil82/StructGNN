from __future__ import annotations

from pathlib import Path

from YJKAPI import *  # type: ignore # noqa: F401,F403

try:
    from .common import setup_file_logger
    from .config import OutputConfig
except ImportError:
    from pipelines.yjk_pipeline.common import setup_file_logger
    from pipelines.yjk_pipeline.config import OutputConfig


class YJKAnalysisRunner:
    """YJK 分析命令执行器。"""

    def __init__(self, output_config: OutputConfig, base_dir: Path):
        self.output_dir = output_config.resolve_output_dir(base_dir)
        self.logger = setup_file_logger(
            "YJKAnalysisRunner",
            self.output_dir / "analysis_runner.log",
            output_config.log_level,
        )
        self.yjks_ui = YJKSUIPy()  # type: ignore
        self.yjks_command = YJKSCommandPy()  # type: ignore

    def _run_command(self, command: str, description: str) -> None:
        self.logger.info("Running: %s", description)
        self.yjks_command.RunCommand(command)

    def run_full_analysis(self) -> bool:
        self.logger.info("Starting full analysis...")

        try:
            self.yjks_ui.QSetCurrentRibbonLabel("IDModule_Axis")
            self.yjks_ui.QSetRunScript(1)

            self._run_command("yjk_repairex", "模型修复")
            self._run_command("yjk_save", "保存模型")
            self._run_command("yjk_setlayersupport", "设置层支撑")
            self._run_command("yjkspre_genmodrel", "生成模型关系")
            self._run_command("yjktransload_tlplan", "传递平面荷载")
            self._run_command("yjktransload_tlvert", "传递竖向荷载")

            self.yjks_ui.QSetCurrentRibbonLabel("IDSPRE_ROOT")
            self._run_command("yjkdesign_dsncalculating_all", "结构计算")
            self.yjks_ui.QSetCurrentRibbonLabel("IDDSN_DSP")

            self.logger.info("Full analysis completed successfully")
            return True

        except Exception as exc:
            self.logger.error("Analysis failed: %s", exc, exc_info=True)
            return False

        finally:
            self.yjks_ui.QSetRunScript(0)
