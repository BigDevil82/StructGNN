"""
读取 case_study 导出的 JSON，并通过 YJK API 建模。

支持两类输入：
1) *_fem_data.json（含 shearwalls / beams）
2) *_shearwall_coords.json（仅剪力墙坐标列表）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from YJKAPI import *  # type: ignore # noqa: F401,F403

Segment = tuple[tuple[float, float], tuple[float, float]]


def _as_point(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"非法点坐标: {value}")
    return float(value[0]), float(value[1])


def _segment_key(
    start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[int, int], tuple[int, int]]:
    p1 = (round(start[0]), round(start[1]))
    p2 = (round(end[0]), round(end[1]))
    return (p1, p2) if p1 <= p2 else (p2, p1)


def _dedup_segments(segments: Iterable[Segment]) -> list[Segment]:
    seen = set()
    result = []
    for start, end in segments:
        key = _segment_key(start, end)
        if key in seen:
            continue
        seen.add(key)
        result.append((start, end))
    return result


def _scale_segment(segment: Segment, coord_scale: float) -> Segment:
    (x1, y1), (x2, y2) = segment
    return ((x1 * coord_scale, y1 * coord_scale), (x2 * coord_scale, y2 * coord_scale))


def _translate_segments_to_origin(
    wall_segments: list[Segment], beam_segments: list[Segment]
) -> tuple[list[Segment], list[Segment], tuple[float, float]]:
    all_points = [point for segment in [*wall_segments, *beam_segments] for point in segment]
    if not all_points:
        return wall_segments, beam_segments, (0.0, 0.0)

    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)

    def _shift(segment: Segment) -> Segment:
        (x1, y1), (x2, y2) = segment
        return ((x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y))

    return (
        [_shift(segment) for segment in wall_segments],
        [_shift(segment) for segment in beam_segments],
        (min_x, min_y),
    )


def load_segments_from_json(json_path: Path) -> tuple[list[Segment], list[Segment]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    wall_segments: list[Segment] = []
    beam_segments: list[Segment] = []

    if isinstance(data, dict):
        if isinstance(data.get("shearwalls"), list):
            for item in data["shearwalls"]:
                wall_segments.append((_as_point(item["start"]), _as_point(item["end"])))
        if isinstance(data.get("beams"), list):
            for item in data["beams"]:
                beam_segments.append((_as_point(item["start"]), _as_point(item["end"])))
        if isinstance(data.get("members"), list):
            for item in data["members"]:
                seg = (_as_point(item["start_coord"]), _as_point(item["end_coord"]))
                if item.get("type") == "shearwall":
                    wall_segments.append(seg)
                elif item.get("type") == "beam":
                    beam_segments.append(seg)
    elif isinstance(data, list):
        # 兼容 *_shearwall_coords.json: [ [[x1,y1],[x2,y2]], ... ]
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            wall_segments.append((_as_point(item[0]), _as_point(item[1])))
    else:
        raise ValueError(f"不支持的 JSON 结构: {type(data)}")

    return _dedup_segments(wall_segments), _dedup_segments(beam_segments)


def _add_wall_or_beam(data_func, std_flr, section, segment: Segment, is_wall: bool):
    (x1, y1), (x2, y2) = segment
    x1_i, y1_i = round(x1), round(y1)
    x2_i, y2_i = round(x2), round(y2)
    if x1_i == x2_i and y1_i == y2_i:
        return

    j1 = data_func.Joint_Generate(std_flr.ID, x1_i, y1_i)
    j2 = data_func.Joint_Generate(std_flr.ID, x2_i, y2_i)
    axis = data_func.Axis_Generate(std_flr.ID, j1.ID, j2.ID)
    grid = data_func.Grid_Generate(std_flr.ID, j1.ID, j2.ID, axis.ID)
    if is_wall:
        data_func.wall_arrange(grid, section)
    else:
        data_func.beam_arrange(grid, section)


def build_model_from_json(
    json_path: Path,
    story_num: int,
    story_height: int,
    wall_thickness: int,
    coord_scale: float,
    with_beam: bool,
    beam_size: str,
    dead_load: float,
    live_load: float,
    refresh_to_yjk: bool,
    ydb_dir: str | None,
    ydb_name: str | None,
):
    wall_segments, beam_segments = load_segments_from_json(json_path)
    if not wall_segments and not (with_beam and beam_segments):
        raise ValueError("JSON 中未找到可建模构件")

    wall_segments = [_scale_segment(segment, coord_scale) for segment in wall_segments]
    beam_segments = [_scale_segment(segment, coord_scale) for segment in beam_segments]

    modeled_beam_segments = beam_segments if with_beam else []
    wall_segments, modeled_beam_segments, translation = _translate_segments_to_origin(
        wall_segments,
        modeled_beam_segments,
    )
    if with_beam:
        beam_segments = modeled_beam_segments

    data_func = DataFunc()  # type: ignore
    std_flr = data_func.StdFlr_Generate(story_height, dead_load, live_load)

    wall_sect = data_func.WallSect_Def(6, 1, wall_thickness)
    beam_sect = data_func.BeamSect_Def(6, 1, beam_size) if with_beam and beam_segments else None

    for seg in wall_segments:
        _add_wall_or_beam(data_func, std_flr, wall_sect, seg, is_wall=True)

    if beam_sect is not None:
        for seg in beam_segments:
            _add_wall_or_beam(data_func, std_flr, beam_sect, seg, is_wall=False)

    if story_num > 1:
        data_func.Floors_Assemb(0, std_flr, story_num, story_height)

    data_func.DbModel_Assign()
    model = data_func.GetDbModelData()
    bridge = Hi_AddToAndReadYjk(model)  # type: ignore

    if ydb_dir:
        out_dir = Path(ydb_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = ydb_name or f"{json_path.stem}.ydb"
        bridge.CreateYDB(str(out_dir), out_name)
        print(f"CreateYDB: {out_dir / out_name}")

    if refresh_to_yjk:
        refresh_ok = bridge.RefreshToYJK()
        print(f"RefreshToYJK: {refresh_ok}")

    print(
        f"建模完成: 剪力墙 {len(wall_segments)} 段, "
        f"梁 {len(beam_segments) if with_beam else 0} 段, "
        f"层数 {story_num}, 层高 {story_height}mm, 坐标缩放系数 {coord_scale}, "
        f"平移量 ({translation[0]:.2f}, {translation[1]:.2f})"
    )


def pyyjks():
    # 直接在这里修改参数（用于在 YJK 中以命令方式调用脚本）
    json_path = (
        r"E:\Common\Desktop\Research\deepLearning\codes\Png2Dxf\result\case_study\archi_comp_fem_data.json"
    )
    story_num = 18
    story_height = 3000
    wall_thickness = 200
    coord_scale = 0.6
    dead_load = 5.0
    live_load = 2.0
    with_beam = True
    beam_size = "300,500"
    no_refresh = False
    ydb_dir = None
    ydb_name = None

    build_model_from_json(
        json_path=Path(json_path),
        story_num=story_num,
        story_height=story_height,
        wall_thickness=wall_thickness,
        coord_scale=coord_scale,
        with_beam=with_beam,
        beam_size=beam_size,
        dead_load=dead_load,
        live_load=live_load,
        refresh_to_yjk=not no_refresh,
        ydb_dir=ydb_dir,
        ydb_name=ydb_name,
    )


if __name__ == "__main__":
    pyyjks()
