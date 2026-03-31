import os
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from data_engine.preprocess.dxf_extractor import DXFExtractor

DEFAULT_LAYOUT_LINE_BUFFER = 20.0
DEFAULT_WALL_THICKNESS = 200.0


def load_layout_reference_geometries(dxf_path: str) -> Dict[str, List[BaseGeometry]]:
    """从 DXF 中读取用于对称检测的 infill_walls 与 rooms 几何体。"""
    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    infill_geometries = [
        _sequence_to_geometry(wall, line_buffer=DEFAULT_LAYOUT_LINE_BUFFER) for wall in extractor.infill_walls
    ]
    room_geometries = [
        _sequence_to_geometry(room.polygon, line_buffer=DEFAULT_LAYOUT_LINE_BUFFER, prefer_polygon=True)
        for room in extractor.rooms
    ]

    return {
        "infill_geometries": [geom for geom in infill_geometries if not geom.is_empty],
        "room_geometries": [geom for geom in room_geometries if not geom.is_empty],
    }


def detect_left_right_symmetry(
    infill_geometries: Sequence[BaseGeometry],
    room_geometries: Sequence[BaseGeometry],
    threshold: float,
) -> Dict[str, object]:
    """检测整体布局是否近似左右对称。"""
    source_name = "rooms"
    reference_geometry = _safe_union(room_geometries)

    if reference_geometry.is_empty:
        source_name = "infill_walls"
        reference_geometry = _safe_union(infill_geometries)

    if reference_geometry.is_empty:
        return {
            "is_symmetric": False,
            "axis_x": None,
            "confidence": 0.0,
            "source": "none",
        }

    minx, _, maxx, _ = reference_geometry.bounds
    axis_x = 0.5 * (minx + maxx)
    mirrored_geometry = mirror_geometry(reference_geometry, axis_x)
    confidence = geometry_iou(reference_geometry, mirrored_geometry)

    return {
        "is_symmetric": confidence >= threshold,
        "axis_x": axis_x,
        "confidence": confidence,
        "source": source_name,
    }


def save_symmetry_detection_debug_plot(
    infill_geometries: Sequence[BaseGeometry],
    room_geometries: Sequence[BaseGeometry],
    symmetry_info: Dict[str, object],
    output_path: str,
):
    """保存布局对称检测调试图。"""
    plt.switch_backend("Agg")
    _ensure_parent_dir(output_path)

    source_name = symmetry_info.get("source", "none")
    if source_name == "rooms":
        reference_geometry = _safe_union(room_geometries)
    elif source_name == "infill_walls":
        reference_geometry = _safe_union(infill_geometries)
    else:
        reference_geometry = GeometryCollection()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axis_x = symmetry_info.get("axis_x")

    if reference_geometry.is_empty or axis_x is None:
        for ax in axes:
            ax.axis("off")
        axes[0].text(0.5, 0.5, "No valid geometry for symmetry detection", ha="center", va="center")
        axes[1].text(0.5, 0.5, f"source={source_name}", ha="center", va="center")
    else:
        mirrored_geometry = mirror_geometry(reference_geometry, axis_x)
        _plot_geometry(axes[0], reference_geometry, facecolor="#8fbcd4", edgecolor="#355c7d", alpha=0.7)
        _draw_axis(axes[0], axis_x, reference_geometry)
        axes[0].set_title(f"Reference Layout ({source_name})")

        _plot_geometry(axes[1], reference_geometry, facecolor="#8fbcd4", edgecolor="#355c7d", alpha=0.35)
        _plot_geometry(axes[1], mirrored_geometry, facecolor="#f2a65a", edgecolor="#b85c38", alpha=0.45)
        _draw_axis(axes[1], axis_x, reference_geometry.union(mirrored_geometry))
        axes[1].set_title(
            "Mirrored Overlay\n"
            f"IoU={symmetry_info['confidence']:.3f}, symmetric={symmetry_info['is_symmetric']}"
        )

        for ax in axes:
            ax.set_aspect("equal")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_symmetry_postprocess_debug_plot(
    raw_walls: Sequence[LineString],
    buildable_lines: Sequence[LineString],
    axis_x: float,
    mode: str,
    output_path: str,
    wall_thickness: float = DEFAULT_WALL_THICKNESS,
    min_length: float = 1.0,
):
    """保存对称化后处理各阶段调试图。"""
    plt.switch_backend("Agg")
    _ensure_parent_dir(output_path)

    half_thickness = wall_thickness / 2.0
    raw_wall_area = _buffer_lines(raw_walls, half_thickness)
    buildable_area = _buffer_lines(buildable_lines, half_thickness)
    mirrored_area = (
        mirror_geometry(raw_wall_area, axis_x) if not raw_wall_area.is_empty else GeometryCollection()
    )

    if mode == "union":
        symmetric_area = raw_wall_area.union(mirrored_area)
    elif mode == "intersection":
        symmetric_area = raw_wall_area.intersection(mirrored_area)
    else:
        symmetric_area = raw_wall_area

    clipped_area = symmetric_area.intersection(buildable_area).buffer(0)
    projected_lines: List[LineString] = []
    for line in buildable_lines:
        projected_lines.extend(_extract_lines(line.intersection(clipped_area), min_length=min_length))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    panels = axes.flatten()

    _plot_lines(panels[0], buildable_lines, color="#9aa0a6", linewidth=2.0)
    _plot_lines(panels[0], raw_walls, color="#d1495b", linewidth=3.0)
    _draw_axis(panels[0], axis_x, buildable_area.union(raw_wall_area))
    panels[0].set_title(f"Raw Walls on Buildable Lines\ncount={len(raw_walls)}")

    _plot_geometry(panels[1], raw_wall_area, facecolor="#d1495b", edgecolor="#8c1c13", alpha=0.6)
    _plot_geometry(panels[1], mirrored_area, facecolor="#edae49", edgecolor="#9c6644", alpha=0.5)
    _draw_axis(panels[1], axis_x, raw_wall_area.union(mirrored_area))
    panels[1].set_title("Buffered Raw vs Mirrored")

    _plot_geometry(panels[2], symmetric_area, facecolor="#00798c", edgecolor="#003d5b", alpha=0.65)
    _draw_axis(panels[2], axis_x, symmetric_area)
    panels[2].set_title(f"Symmetric Area ({mode})")

    _plot_geometry(panels[3], buildable_area, facecolor="#c8d5b9", edgecolor="#588157", alpha=0.45)
    _plot_geometry(panels[3], clipped_area, facecolor="#d1495b", edgecolor="#8c1c13", alpha=0.6)
    _draw_axis(panels[3], axis_x, buildable_area.union(clipped_area))
    panels[3].set_title("Clipped by Buildable Area")

    _plot_lines(panels[4], buildable_lines, color="#9aa0a6", linewidth=2.0)
    _plot_lines(panels[4], projected_lines, color="#2a9d8f", linewidth=3.0)
    _draw_axis(panels[4], axis_x, buildable_area)
    panels[4].set_title(f"Projected Symmetric Lines\ncount={len(projected_lines)}")

    _plot_lines(panels[5], raw_walls, color="#d1495b", linewidth=3.0, alpha=0.75)
    _plot_lines(panels[5], projected_lines, color="#2a9d8f", linewidth=2.0, alpha=0.9)
    _draw_axis(panels[5], axis_x, buildable_area.union(raw_wall_area).union(clipped_area))
    panels[5].set_title("Before / After Overlay")

    for ax in panels:
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def symmetrize_raw_walls(
    raw_walls: Sequence[LineString],
    buildable_lines: Sequence[LineString],
    axis_x: float,
    mode: str,
    wall_thickness: float = DEFAULT_WALL_THICKNESS,
    min_length: float = 1.0,
    debug_path: str = None,
) -> List[LineString]:
    """对原始剪力墙线段做左右对称后处理，并重新投影回可布置边界。"""
    if mode == "none" or not raw_walls:
        return list(raw_walls)

    if mode not in {"union", "intersection"}:
        raise ValueError(f"Unsupported symmetry mode: {mode}")

    half_thickness = wall_thickness / 2.0
    raw_wall_area = _buffer_lines(raw_walls, half_thickness)
    buildable_area = _buffer_lines(buildable_lines, half_thickness)

    if raw_wall_area.is_empty or buildable_area.is_empty:
        return list(raw_walls)

    mirrored_area = mirror_geometry(raw_wall_area, axis_x)
    if mode == "union":
        symmetric_area = raw_wall_area.union(mirrored_area)
    else:
        symmetric_area = raw_wall_area.intersection(mirrored_area)

    clipped_area = symmetric_area.intersection(buildable_area).buffer(0)
    if clipped_area.is_empty:
        return []

    projected_lines: List[LineString] = []
    for line in buildable_lines:
        intersection = line.intersection(clipped_area)
        projected_lines.extend(_extract_lines(intersection, min_length=min_length))

    if debug_path:
        save_symmetry_postprocess_debug_plot(
            raw_walls=raw_walls,
            buildable_lines=buildable_lines,
            axis_x=axis_x,
            mode=mode,
            output_path=debug_path,
            wall_thickness=wall_thickness,
            min_length=min_length,
        )

    return projected_lines


def mirror_geometry(geometry: BaseGeometry, axis_x: float) -> BaseGeometry:
    """关于 x=axis_x 做镜像。"""
    return affinity.scale(geometry, xfact=-1.0, yfact=1.0, origin=(axis_x, 0.0))


def geometry_iou(geometry_a: BaseGeometry, geometry_b: BaseGeometry, eps: float = 1e-6) -> float:
    """基于面积计算两个几何体的 IoU。"""
    union = geometry_a.union(geometry_b)
    union_area = union.area
    if union.is_empty or union_area <= eps:
        return 0.0
    intersection_area = geometry_a.intersection(geometry_b).area
    return float((intersection_area + eps) / (union_area + eps))


def _buffer_lines(lines: Sequence[LineString], distance: float) -> BaseGeometry:
    buffered = [
        line.buffer(distance, cap_style=2, join_style=2)
        for line in lines
        if line is not None and not line.is_empty and line.length > 0
    ]
    return _safe_union(buffered)


def _safe_union(geometries: Iterable[BaseGeometry]) -> BaseGeometry:
    valid = [geom.buffer(0) for geom in geometries if geom is not None and not geom.is_empty]
    if not valid:
        return GeometryCollection()
    return unary_union(valid).buffer(0)


def _extract_lines(geometry: BaseGeometry, min_length: float) -> List[LineString]:
    if geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        return [geometry] if geometry.length >= min_length else []

    if isinstance(geometry, MultiLineString):
        return [line for line in geometry.geoms if line.length >= min_length]

    if hasattr(geometry, "geoms"):
        lines: List[LineString] = []
        for geom in geometry.geoms:
            lines.extend(_extract_lines(geom, min_length=min_length))
        return lines

    return []


def _sequence_to_geometry(
    points: Sequence,
    line_buffer: float,
    prefer_polygon: bool = False,
) -> BaseGeometry:
    coords = _normalize_coords(points)
    if len(coords) < 2:
        return GeometryCollection()

    if len(coords) >= 3 and prefer_polygon:
        polygon = Polygon(coords)
        polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-3:
            return polygon

    if len(coords) >= 3:
        polygon = Polygon(coords)
        polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-3:
            return polygon

    line = LineString(coords)
    if line.is_empty or line.length <= 0:
        return GeometryCollection()
    return line.buffer(line_buffer, cap_style=2, join_style=2)


def _normalize_coords(points: Sequence) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for point in points:
        if isinstance(point, dict):
            x_val = point.get("X")
            y_val = point.get("Y")
        else:
            x_val = getattr(point, "x", None)
            y_val = getattr(point, "y", None)

        if x_val is None or y_val is None:
            continue
        coords.append((float(x_val), float(y_val)))

    if len(coords) >= 2 and coords[0] == coords[-1]:
        return coords[:-1]
    return coords


def _ensure_parent_dir(output_path: str):
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _plot_lines(ax, lines: Sequence[LineString], color: str, linewidth: float, alpha: float = 1.0):
    for line in lines:
        if line is None or line.is_empty:
            continue
        x_coords, y_coords = line.xy
        ax.plot(x_coords, y_coords, color=color, linewidth=linewidth, alpha=alpha)


def _plot_geometry(ax, geometry: BaseGeometry, facecolor: str, edgecolor: str, alpha: float):
    if geometry is None or geometry.is_empty:
        return

    if isinstance(geometry, Polygon):
        x_coords, y_coords = geometry.exterior.xy
        ax.fill(x_coords, y_coords, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
        for interior in geometry.interiors:
            hole_x, hole_y = interior.xy
            ax.fill(hole_x, hole_y, facecolor="white", edgecolor=edgecolor, alpha=1.0)
        return

    if hasattr(geometry, "geoms"):
        for geom in geometry.geoms:
            _plot_geometry(ax, geom, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)


def _draw_axis(ax, axis_x: float, geometry: BaseGeometry):
    if geometry is None or geometry.is_empty:
        return
    minx, miny, maxx, maxy = geometry.bounds
    span = max(maxy - miny, 1.0)
    padding = 0.05 * span
    ax.plot([axis_x, axis_x], [miny - padding, maxy + padding], linestyle="--", color="black", linewidth=1.2)
