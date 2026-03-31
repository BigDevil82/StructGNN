from __future__ import annotations

from pathlib import Path

from YJKAPI import *  # type: ignore # noqa: F401,F403

try:
    from .common import setup_file_logger
    from .config import ModelConfig, OutputConfig
    from .json_loader import Segment
except ImportError:
    from common import setup_file_logger
    from config import ModelConfig, OutputConfig
    from json_loader import Segment


class YJKJsonModelBuilder:
    """根据线段数据构建 YJK 模型。"""

    def __init__(self, model_config: ModelConfig, output_config: OutputConfig, base_dir: Path):
        self.model_config = model_config
        self.output_config = output_config
        self.output_dir = output_config.resolve_output_dir(base_dir)
        self.logger = setup_file_logger(
            "YJKJsonModelBuilder",
            self.output_dir / "model_builder.log",
            output_config.log_level,
        )
        self.data_func = DataFunc()  # type: ignore

    def _build_standard_floor_para(self) -> str:
        para = BzcPara()  # type: ignore

        para.setVal(1, self.model_config.slab_thickness)
        para.setVal(2, 350)
        para.setVal(3, 15)
        para.setVal(23, self.model_config.rebar_grade)

        para.setVal(4, 400)
        para.setVal(8, self.model_config.rebar_grade)
        para.setVal(14, 30)
        para.setVal(16, self.model_config.rebar_grade)

        para.setVal(5, 350)
        para.setVal(7, self.model_config.rebar_grade)
        para.setVal(17, self.model_config.rebar_grade)

        para.setVal(6, 400)
        para.setVal(12, self.model_config.rebar_grade)
        para.setVal(18, self.model_config.rebar_grade)
        para.setVal(19, self.model_config.rebar_grade)
        para.setVal(20, self.model_config.rebar_grade)
        para.setVal(22, 20)

        return para.GetValString()

    def _set_analysis_parameters(self) -> None:
        self.data_func.ProjectPara_Set(601, "60102")
        self.data_func.ProjectPara_Set(603, "60302")
        self.data_func.ProjectPara_Set(604, "60402")

    @staticmethod
    def _add_member_segment(data_func, std_flr, section, segment: Segment, is_wall: bool) -> bool:
        (x1, y1), (x2, y2) = segment
        x1_i, y1_i = round(x1), round(y1)
        x2_i, y2_i = round(x2), round(y2)

        if x1_i == x2_i and y1_i == y2_i:
            return False

        j1 = data_func.Joint_Generate(std_flr.ID, x1_i, y1_i)
        j2 = data_func.Joint_Generate(std_flr.ID, x2_i, y2_i)
        axis = data_func.Axis_Generate(std_flr.ID, j1.ID, j2.ID)
        grid = data_func.Grid_Generate(std_flr.ID, j1.ID, j2.ID, axis.ID)

        if is_wall:
            data_func.wall_arrange(grid, section)
        else:
            data_func.beam_arrange(grid, section)

        return True

    def build_model(
        self,
        wall_segments: list[Segment],
        beam_segments: list[Segment],
    ) -> bool:
        """构建并下发模型。"""
        self.logger.info("Starting model construction from JSON segments...")

        if not wall_segments and not (self.model_config.with_beam and beam_segments):
            raise ValueError("无可建模构件")

        try:
            self._set_analysis_parameters()

            std_flr = self.data_func.StdFlr_Generate(
                self.model_config.story_height,
                self.model_config.slab_dead_load,
                self.model_config.slab_live_load,
                self._build_standard_floor_para(),
                self.model_config.std_floor_name,
            )

            wall_sect = self.data_func.WallSect_Def(
                6,
                1,
                self.model_config.wall_thickness,
                f"QW{self.model_config.wall_thickness}_C40",
            )

            beam_sect = None
            if self.model_config.with_beam and beam_segments:
                beam_sect = self.data_func.BeamSect_Def(6, 1, self.model_config.beam_size)

            wall_count = 0
            for segment in wall_segments:
                if self._add_member_segment(self.data_func, std_flr, wall_sect, segment, is_wall=True):
                    wall_count += 1

            beam_count = 0
            if beam_sect is not None:
                for segment in beam_segments:
                    if self._add_member_segment(
                        self.data_func,
                        std_flr,
                        beam_sect,
                        segment,
                        is_wall=False,
                    ):
                        beam_count += 1

            if self.model_config.story_num > 1:
                self.data_func.Floors_Assemb(
                    0,
                    std_flr,
                    self.model_config.story_num,
                    self.model_config.story_height,
                )

            self.data_func.DbModel_Assign()
            model = self.data_func.GetDbModelData()
            bridge = Hi_AddToAndReadYjk(model)  # type: ignore

            if self.output_config.ydb_dir:
                ydb_dir = Path(self.output_config.ydb_dir).resolve()
                ydb_dir.mkdir(parents=True, exist_ok=True)
                ydb_name = self.output_config.ydb_name or "model_from_json.ydb"
                bridge.CreateYDB(str(ydb_dir), ydb_name)
                self.logger.info("CreateYDB: %s", ydb_dir / ydb_name)

            refresh_ok = True
            if self.output_config.refresh_to_yjk:
                refresh_ok = bool(bridge.RefreshToYJK())
                self.logger.info("RefreshToYJK: %s", refresh_ok)

            self.logger.info(
                "Model build finished. walls=%s, beams=%s, story_num=%s, story_height=%s",
                wall_count,
                beam_count,
                self.model_config.story_num,
                self.model_config.story_height,
            )
            return refresh_ok

        except Exception as exc:
            self.logger.error("Model build failed: %s", exc, exc_info=True)
            return False
