import math
from pathlib import Path
from typing import Union

from .domain import BeamRole, FEMInput


def _polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def _segment_length(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return (dx * dx + dy * dy) ** 0.5


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2 == 1:
        return vs[mid]
    return 0.5 * (vs[mid - 1] + vs[mid])


def _rect_slab_span_depth(slab: list[tuple[float, float]]) -> Union[tuple[float, float], None]:
    if len(slab) != 4:
        return None

    edges = [_segment_length(slab[i], slab[(i + 1) % 4]) for i in range(4)]
    if any(e <= 1.0e-9 for e in edges):
        return None

    # Opposite edges of a rectangle should be close. Keep a loose tolerance for noisy CAD input.
    tol = 0.20
    e0, e1, e2, e3 = edges
    if abs(e0 - e2) / max(e0, e2) > tol:
        return None
    if abs(e1 - e3) / max(e1, e3) > tol:
        return None

    span = min(edges)
    depth = max(edges)
    return span, depth


def _filter_reasonable_rect_slabs(rect_dims: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(rect_dims) <= 2:
        return rect_dims

    spans = [d[0] for d in rect_dims]
    depths = [d[1] for d in rect_dims]
    areas = [d[0] * d[1] for d in rect_dims]
    span_med = _median(spans)
    depth_med = _median(depths)
    area_med = _median(areas)

    # Remove tiny outliers (typically shafts/void artifacts) while preserving normal rooms.
    span_min = max(0.8, 0.55 * span_med)
    depth_min = max(1.0, 0.55 * depth_med)
    area_min = max(2.0, 0.35 * area_med)

    filtered: list[tuple[float, float]] = []
    for span, depth in rect_dims:
        area = span * depth
        aspect = depth / max(span, 1.0e-9)
        if span < span_min or depth < depth_min or area < area_min:
            continue
        if aspect > 3.5:
            continue
        if span < 2.0 or depth < 3.0:
            continue
        if span > 8.0 or depth > 12.0:
            continue
        filtered.append((span, depth))

    return filtered if filtered else rect_dims


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_scale_factor(input_data: FEMInput, low: float = 2.0, high: float = 6.0, seed: int = 42) -> float:
    _ = seed  # keep signature backward compatible; scaling is deterministic now.

    if low <= 0.0 or high <= 0.0 or high < low:
        raise ValueError("choose_scale_factor expects 0 < low <= high.")

    rect_dims = [sd for slab in input_data.slabs if (sd := _rect_slab_span_depth(slab)) is not None]

    rect_dims = _filter_reasonable_rect_slabs(rect_dims)
    spans = [d[0] for d in rect_dims]
    depths = [d[1] for d in rect_dims]

    wall_lengths = [m.length for m in input_data.walls]
    beam_lengths = [m.length for m in input_data.beams]

    weighted_scales: list[tuple[float, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    # Residential room span/depth priors (meters) based on common apartment bays.
    span_target = 3.3
    depth_target = 4.5
    span_min, span_max = 2.7, 4.2
    depth_min, depth_max = 3.3, 6.0

    if spans:
        span_med = _median(spans)
        weighted_scales.append((span_target / span_med, 0.45))
        lower_bounds.append(span_min / span_med)
        upper_bounds.append(span_max / span_med)

    if depths:
        depth_med = _median(depths)
        # print(
        #     f"Median rectangular slab span: {span_med:.3f} m, depth: {depth_med:.3f} m (count={len(rect_dims)})"
        # )
        weighted_scales.append((depth_target / depth_med, 0.35))
        lower_bounds.append(depth_min / depth_med)
        upper_bounds.append(depth_max / depth_med)

    # Longest wall and beam constraints keep the global plan size in a practical interval.
    target_long = 0.5 * (low + high)
    if wall_lengths:
        wall_max = max(wall_lengths)
        weighted_scales.append((target_long / wall_max, 0.12))
        lower_bounds.append(low / wall_max)
        upper_bounds.append(high / wall_max)
    if beam_lengths:
        beam_max = max(beam_lengths)
        weighted_scales.append((target_long / beam_max, 0.08))
        lower_bounds.append(low / beam_max)
        upper_bounds.append(high / beam_max)

    if not weighted_scales:
        raise ValueError("Cannot infer scale factor: no usable slab, wall, or beam geometry.")

    # Weighted geometric mean is robust to different scale magnitudes.
    total_w = sum(w for _, w in weighted_scales)
    log_sum = 0.0
    for s, w in weighted_scales:
        log_sum += w * math.log(s)
    scale = math.exp(log_sum / total_w)

    if lower_bounds and upper_bounds:
        lb = max(lower_bounds)
        ub = min(upper_bounds)
        if lb <= ub:
            scale = _clamp(scale, lb, ub)

    return scale


def estimate_floor_area(input_data: FEMInput) -> float:
    slab_areas = [_polygon_area(slab) for slab in input_data.slabs if len(slab) >= 3]
    slab_area = sum(a for a in slab_areas if a > 0.0)
    if slab_area > 1.0e-9:
        # print(f"Floor area from slabs: {slab_area:.3f} m^2 (count={len(slab_areas)})")
        return slab_area

    points = []
    for m in input_data.all_members():
        points.append(m.start)
        points.append(m.end)
    if len(points) < 2:
        raise ValueError("Cannot estimate floor area: no valid geometry points.")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lx = max(xs) - min(xs)
    ly = max(ys) - min(ys)
    print(f"Estimated floor plan bounding box: {lx:.3f} m x {ly:.3f} m")
    area = lx * ly
    return area


def estimate_structural_self_mass_per_floor(
    input_data: FEMInput,
    story_height: float,
    wall_thickness: float,
    primary_beam_width: float,
    primary_beam_depth: float,
    secondary_beam_width: float,
    secondary_beam_depth: float,
    slab_thickness: float,
    density_kg_m3: float,
    floor_area: Union[float, None] = None,
) -> dict[str, float]:
    area = estimate_floor_area(input_data) if floor_area is None else floor_area

    wall_length_total = sum(m.length for m in input_data.walls)
    primary_beam_length_total = input_data.beam_length_by_role(BeamRole.PRIMARY)
    secondary_beam_length_total = input_data.beam_length_by_role(BeamRole.SECONDARY)

    wall_vol = wall_length_total * wall_thickness * story_height
    primary_beam_vol = primary_beam_length_total * primary_beam_width * primary_beam_depth
    secondary_beam_vol = secondary_beam_length_total * secondary_beam_width * secondary_beam_depth
    beam_vol = primary_beam_vol + secondary_beam_vol
    slab_vol = area * slab_thickness

    wall_mass = wall_vol * density_kg_m3
    primary_beam_mass = primary_beam_vol * density_kg_m3
    secondary_beam_mass = secondary_beam_vol * density_kg_m3
    beam_mass = beam_vol * density_kg_m3
    slab_mass = slab_vol * density_kg_m3
    total_mass = wall_mass + beam_mass + slab_mass

    return {
        "wall_mass": wall_mass,
        "primary_beam_mass": primary_beam_mass,
        "secondary_beam_mass": secondary_beam_mass,
        "beam_mass": beam_mass,
        "slab_mass": slab_mass,
        "total_mass": total_mass,
    }


def load_and_scale_input(
    json_path: Path,
    input_unit_scale_to_m: float,
    enable_auto_scale: bool,
    low: float,
    high: float,
    seed: int,
    manual_factor: Union[float, None],
) -> tuple[FEMInput, float]:
    input_data = FEMInput.from_json(json_path, xy_scale_to_m=input_unit_scale_to_m)
    if not input_data.all_members():
        raise ValueError("No beams/walls found in JSON.")

    if manual_factor is not None:
        factor = manual_factor
    elif enable_auto_scale:
        factor = choose_scale_factor(input_data, low=low, high=high, seed=seed)
    else:
        factor = 1.0

    return input_data.scaled(factor), factor
