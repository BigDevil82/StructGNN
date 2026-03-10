"""
CAD inference pipeline:
1) Parse CAD JSON (wall/door/window lines) and extract wall centerlines + rooms.
2) Convert results to DXF in the same layer format as preprocess.dxf_extractor.
3) Build inference graph data with door/window-aware masks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import ezdxf
import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from axis_engine.cad_processor import CADLayoutProcessor
from preprocess.layout_graph import LayoutGraphBuilder
from preprocess.room_analyzer import RoomAnalyzer
from preprocess.room_calibrator import calibrate_rooms


def _ensure_layer(doc, name: str, color: int) -> None:
    if name in doc.layers:
        doc.layers.get(name).dxf.color = color
        return
    doc.layers.new(name=name, dxfattribs={"color": color})


def _poly_exterior_points(poly: Polygon) -> List[Tuple[float, float]]:
    coords = list(poly.exterior.coords)
    if len(coords) < 4:
        return []
    if coords[0] == coords[-1]:
        coords = coords[:-1]
    return [(float(x), float(y)) for x, y in coords]


def _flatten_polygons(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [g for g in geom.geoms if isinstance(g, Polygon)]
    return []


def _buffer_line(line: LineString, width: float) -> Optional[Polygon]:
    if line.length < 1e-3 or width <= 0:
        return None
    poly = line.buffer(width / 2.0, cap_style=2, join_style=2)
    if poly.is_empty:
        return None
    if isinstance(poly, Polygon):
        return poly
    if isinstance(poly, MultiPolygon):
        return max(poly.geoms, key=lambda p: p.area) if poly.geoms else None
    return None


def _buffer_lines(geom, width: float) -> List[Polygon]:
    if geom is None:
        return []
    polys: List[Polygon] = []
    if isinstance(geom, LineString):
        poly = _buffer_line(geom, width)
        if poly:
            polys.append(poly)
        return polys
    if isinstance(geom, MultiLineString):
        for line in geom.geoms:
            poly = _buffer_line(line, width)
            if poly:
                polys.append(poly)
    return polys


def _points_to_polygon(points: Sequence, fallback_width: Optional[float] = None) -> Optional[Polygon]:
    coords = [(float(p.x), float(p.y)) for p in points]
    if len(coords) >= 3:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    if len(coords) == 2 and fallback_width:
        line = LineString(coords)
        return _buffer_line(line, fallback_width)
    return None


def write_inference_dxf(
    dxf_path: str,
    infill_walls: Iterable[Polygon],
    doors: Iterable[Polygon],
    windows: Iterable[Polygon],
    rooms: Iterable[Polygon],
    shear_walls: Optional[Iterable[Polygon]] = None,
) -> None:
    doc = ezdxf.new(setup=True)
    msp = doc.modelspace()

    layer_colors = {
        "SHEAR_WALLS": 1,
        "INFILL_WALLS": 8,
        "DOORS": 5,
        "WINDOWS": 3,
        "ROOM": 4,
    }
    for name, color in layer_colors.items():
        _ensure_layer(doc, name, color)

    def add_polys(polys: Iterable[Polygon], layer: str) -> None:
        for poly in polys:
            if poly is None or poly.is_empty:
                continue
            for p in _flatten_polygons(poly):
                points = _poly_exterior_points(p)
                if len(points) < 3:
                    continue
                msp.add_lwpolyline(points, close=True, dxfattribs={"layer": layer})

    if shear_walls:
        add_polys(shear_walls, "SHEAR_WALLS")
    add_polys(infill_walls, "INFILL_WALLS")
    add_polys(doors, "DOORS")
    add_polys(windows, "WINDOWS")
    add_polys(rooms, "ROOM")

    doc.saveas(dxf_path)


def convert_cad_json_to_dxf(
    json_path: str,
    dxf_path: str,
    *,
    default_wall_thickness: float = 200.0,
    door_width: float = 200.0,
    window_width: float = 200.0,
    min_wall_length: float = 50.0,
    shear_threshold: Optional[float] = None,
) -> str:
    processor = CADLayoutProcessor(json_path)
    processor.build_geometry()
    processor.extract_centerlines()
    processor.generate_rooms()

    thicknesses = [t for _, t in processor.wall_centerlines if t and t > 0]
    if thicknesses:
        default_wall_thickness = float(np.median(thicknesses))

    infill_polys: List[Polygon] = []
    shear_polys: List[Polygon] = []

    for line, thickness in processor.wall_centerlines:
        if line.length < min_wall_length:
            continue
        t = thickness if thickness and thickness > 0 else default_wall_thickness
        poly = _buffer_line(line, t)
        if not poly:
            continue
        if shear_threshold is not None and t >= shear_threshold:
            shear_polys.append(poly)
        else:
            infill_polys.append(poly)

    doors = _buffer_lines(processor.components.get("doors"), door_width)
    windows = _buffer_lines(processor.components.get("windows"), window_width)
    rooms = list(processor.rooms) if hasattr(processor, "rooms") else []

    write_inference_dxf(
        dxf_path,
        infill_walls=infill_polys,
        doors=doors,
        windows=windows,
        rooms=rooms,
        shear_walls=shear_polys if shear_polys else None,
    )

    return dxf_path


def build_graph_from_dxf_inference(
    dxf_path: str,
    *,
    opening_buffer: float = 100.0,
    mode: str = "none",
):
    from preprocess.dxf_extractor import DXFExtractor
    from shearwall_pred.augmentor import GeometryAugmentor

    extractor = DXFExtractor()
    extractor.extract_from_file(dxf_path)

    raw_rooms = [Polygon([(p.x, p.y) for p in r.polygon]) for r in extractor.rooms]
    infill_polys = [Polygon([(p.x, p.y) for p in w]) for w in extractor.infill_walls]

    door_polys = [
        _points_to_polygon(w, fallback_width=opening_buffer) for w in extractor.doors if w is not None
    ]
    window_polys = [
        _points_to_polygon(w, fallback_width=opening_buffer) for w in extractor.windows if w is not None
    ]
    openings = [p for p in door_polys + window_polys if p is not None and not p.is_empty]

    if openings:
        openings_union = unary_union(openings).buffer(0)
        buildable_polys: List[Polygon] = []
        for poly in infill_polys:
            if poly.is_empty:
                continue
            diff = poly.difference(openings_union)
            buildable_polys.extend(_flatten_polygons(diff))
    else:
        buildable_polys = infill_polys

    if not raw_rooms:
        raise ValueError(f"No valid rooms found in DXF: {dxf_path}")

    if mode and mode != "none":
        raw_rooms, _, buildable_polys = GeometryAugmentor.apply_augmentation(
            raw_rooms, [], buildable_polys, mode
        )

    calibrated_rooms = calibrate_rooms(raw_rooms, alignment_threshold=200.0)
    analyzer = RoomAnalyzer(sw_polys=[], infill_polys=buildable_polys)

    analysis_results = []
    for i, room in enumerate(raw_rooms):
        if not room.is_valid or room.area < 1:
            continue
        sw_vector, masks = analyzer.process_room(room)
        analysis_results.append({"room_index": i, "sw_vector": sw_vector, "masks": masks})

    graph_builder = LayoutGraphBuilder(dxf_path, calibrated_rooms)
    graph_builder.add_analysis_results(analysis_results)
    return graph_builder


def build_graph_from_cad_json(
    json_path: str,
    dxf_output: Optional[str] = None,
    *,
    mode: str = "none",
    door_width: float = 100.0,
    window_width: float = 80.0,
    opening_buffer: float = 100.0,
) -> LayoutGraphBuilder:
    dxf_path = dxf_output
    if dxf_path is None:
        dxf_path = str(Path(json_path).with_suffix(".dxf"))

    convert_cad_json_to_dxf(
        json_path,
        dxf_path,
        door_width=door_width,
        window_width=window_width,
        shear_threshold=None,
    )
    return build_graph_from_dxf_inference(dxf_path, opening_buffer=opening_buffer, mode=mode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CAD JSON to DXF for inference.")
    parser.add_argument("--cad-json", required=True, help="Input CAD JSON path")
    parser.add_argument("--dxf-out", required=True, help="Output DXF path")
    parser.add_argument("--door-width", type=float, default=200.0, help="Door buffer width (mm)")
    parser.add_argument("--window-width", type=float, default=200.0, help="Window buffer width (mm)")
    parser.add_argument("--wall-threshold", type=float, default=None, help="Optional shear threshold (mm)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    convert_cad_json_to_dxf(
        args.cad_json,
        args.dxf_out,
        door_width=args.door_width,
        window_width=args.window_width,
        shear_threshold=args.wall_threshold,
    )


if __name__ == "__main__":
    main()
