import math
from pathlib import Path
from typing import Union

from ..core.constants import RectSlabFilterBounds
from ..core.domain import FEMInput
from .mass_estimation import estimate_floor_area, estimate_structural_self_mass_per_floor
from .primitives import polygon_area, rect_slab_span_depth


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    n = len(vs)
    mid = n // 2
    if n % 2 == 1:
        return vs[mid]
    return 0.5 * (vs[mid - 1] + vs[mid])


def _resolve_rect_slab_thresholds(
    rect_dims: list[tuple[float, float]], bounds: RectSlabFilterBounds
) -> tuple[float, float, float]:
    spans = [span for span, _ in rect_dims]
    depths = [depth for _, depth in rect_dims]
    areas = [span * depth for span, depth in rect_dims]
    span_med = _median(spans)
    depth_med = _median(depths)
    area_med = _median(areas)
    return (
        max(bounds.min_span_floor, bounds.median_span_factor * span_med),
        max(bounds.min_depth_floor, bounds.median_depth_factor * depth_med),
        max(bounds.min_area_floor, bounds.median_area_factor * area_med),
    )


def _is_reasonable_rect_slab(
    span: float,
    depth: float,
    span_min: float,
    depth_min: float,
    area_min: float,
    bounds: RectSlabFilterBounds,
) -> bool:
    area = span * depth
    aspect = depth / max(span, 1.0e-9)
    return (
        span >= span_min
        and depth >= depth_min
        and area >= area_min
        and span >= bounds.min_span
        and span <= bounds.max_span
        and depth >= bounds.min_depth
        and depth <= bounds.max_depth
        and aspect <= bounds.max_aspect_ratio
    )


def _filter_reasonable_rect_slabs(rect_dims: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(rect_dims) <= 2:
        return rect_dims

    bounds = RectSlabFilterBounds()
    span_min, depth_min, area_min = _resolve_rect_slab_thresholds(rect_dims, bounds)
    filtered = [
        (span, depth)
        for span, depth in rect_dims
        if _is_reasonable_rect_slab(span, depth, span_min, depth_min, area_min, bounds)
    ]

    return filtered if filtered else rect_dims


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def choose_scale_factor(input_data: FEMInput, low: float = 2.0, high: float = 6.0, seed: int = 42) -> float:
    _ = seed  # keep signature backward compatible; scaling is deterministic now.

    if low <= 0.0 or high <= 0.0 or high < low:
        raise ValueError("choose_scale_factor expects 0 < low <= high.")

    rect_dims: list[tuple[float, float]] = []
    for slab in input_data.slabs:
        if not isinstance(slab, list):
            continue
        sd = rect_slab_span_depth(slab)
        if sd is None:
            continue
        rect_dims.append(sd)

    rect_dims = _filter_reasonable_rect_slabs(rect_dims)
    spans = [d[0] for d in rect_dims]
    depths = [d[1] for d in rect_dims]

    wall_lengths = [m.length for m in input_data.walls if m.length > 1.0e-9]
    beam_lengths = [m.length for m in input_data.beams if m.length > 1.0e-9]

    weighted_scales: list[tuple[float, float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    # Residential room span/depth priors (meters) based on common apartment bays.
    span_target = 3.3
    depth_target = 4.5
    span_min, span_max = 2.7, 4.2
    depth_min, depth_max = 3.3, 6.0

    if spans and depths:
        span_med = _median(spans)
        depth_med = _median(depths)
        # print(
        #     f"Median rectangular slab span: {span_med:.3f} m, depth: {depth_med:.3f} m (count={len(rect_dims)})"
        # )
        if span_med > 1.0e-9:
            weighted_scales.append((span_target / span_med, 0.45))
            lower_bounds.append(span_min / span_med)
            upper_bounds.append(span_max / span_med)
        if depth_med > 1.0e-9:
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
        return 1.0

    # Weighted geometric mean is robust to different scale magnitudes.
    total_w = sum(w for _, w in weighted_scales)
    log_sum = 0.0
    for s, w in weighted_scales:
        log_sum += w * (0.0 if s <= 1.0e-12 else math.log(s))
    scale = math.exp(log_sum / max(total_w, 1.0e-12))

    if lower_bounds and upper_bounds:
        lb = max(lower_bounds)
        ub = min(upper_bounds)
        if lb <= ub:
            scale = _clamp(scale, lb, ub)
        else:
            # If constraints conflict due to noisy geometry, use the closest feasible side.
            scale = lb if scale < lb else ub

    return max(scale, 1.0e-6)


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
