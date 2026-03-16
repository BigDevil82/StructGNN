from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

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


def dedup_segments(segments: Iterable[Segment]) -> list[Segment]:
    seen = set()
    result = []
    for start, end in segments:
        key = _segment_key(start, end)
        if key in seen:
            continue
        seen.add(key)
        result.append((start, end))
    return result


def load_segments_from_json(json_path: Path) -> tuple[list[Segment], list[Segment]]:
    """加载剪力墙与梁线段。"""
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
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            wall_segments.append((_as_point(item[0]), _as_point(item[1])))
    else:
        raise ValueError(f"不支持的 JSON 结构: {type(data)}")

    return dedup_segments(wall_segments), dedup_segments(beam_segments)


def scale_segments(segments: Iterable[Segment], coord_scale: float) -> list[Segment]:
    if coord_scale <= 0:
        raise ValueError("coord_scale 必须为正数")

    out = []
    for (x1, y1), (x2, y2) in segments:
        out.append(((x1 * coord_scale, y1 * coord_scale), (x2 * coord_scale, y2 * coord_scale)))
    return out


def translate_segments_to_origin(
    wall_segments: list[Segment],
    beam_segments: list[Segment],
) -> tuple[list[Segment], list[Segment], tuple[float, float]]:
    all_points = [p for seg in [*wall_segments, *beam_segments] for p in seg]
    if not all_points:
        return wall_segments, beam_segments, (0.0, 0.0)

    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)

    def shift(seg: Segment) -> Segment:
        (x1, y1), (x2, y2) = seg
        return ((x1 - min_x, y1 - min_y), (x2 - min_x, y2 - min_y))

    return [shift(seg) for seg in wall_segments], [shift(seg) for seg in beam_segments], (min_x, min_y)
