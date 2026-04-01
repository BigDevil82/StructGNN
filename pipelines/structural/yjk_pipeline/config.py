from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """YJK 建模参数"""

    story_height: int = 3000
    story_num: int = 20

    slab_thickness: int = 120
    slab_dead_load: float = 5.0
    slab_live_load: float = 2.0

    wall_thickness: int = 200
    beam_size: str = "300,500"

    rebar_grade: int = 3
    std_floor_name: str = "剪力墙标准层"

    coord_scale: float = 1.0
    normalize_to_origin: bool = True
    with_beam: bool = True

    json_path: Optional[Path] = (
        "E:/Common/Desktop/Research/deepLearning/codes/Png2Dxf/outputs/result/case_study/building_components_fem_data.json"
    )


@dataclass
class AnalysisConfig:
    """YJK 分析执行参数"""

    run_analysis: bool = True


@dataclass
class ExtractionConfig:
    """分析结果提取参数"""

    seis_module: int = 4
    enable_total_mass_api: bool = False
    export_plot: bool = True


@dataclass
class OutputConfig:
    """输出控制参数"""

    output_dir: Optional[Path] = "yjk_pipeline_output"
    clean_old_files: bool = True
    refresh_to_yjk: bool = True
    ydb_dir: Optional[Path] = None
    ydb_name: Optional[str] = None
    log_level: int = logging.INFO

    def resolve_output_dir(self, base_dir: Path) -> Path:
        out_dir = Path(self.output_dir) if self.output_dir else (base_dir / "output")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir


@dataclass
class PipelineConfig:
    """统一流程配置"""

    model: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
