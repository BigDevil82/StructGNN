import random
from pathlib import Path
from typing import Union

from .domain import FEMInput


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


def choose_scale_factor(input_data: FEMInput, low: float = 6.0, high: float = 12.0, seed: int = 42) -> float:
    lengths = [m.length for m in input_data.all_members() if m.length > 1e-9]
    if not lengths:
        return 1.0
    max_len = max(lengths)
    rng = random.Random(seed)
    target_max = rng.uniform(low, high)
    return target_max / max_len


def estimate_floor_area(input_data: FEMInput) -> float:
    slab_areas = [_polygon_area(slab) for slab in input_data.slabs if len(slab) >= 3]
    slab_area = sum(a for a in slab_areas if a > 0.0)
    if slab_area > 1.0e-9:
        print(f"Floor area from slabs: {slab_area:.3f} m^2 (count={len(slab_areas)})")
        return max(slab_area, 16.0)

    points = []
    for m in input_data.all_members():
        points.append(m.start)
        points.append(m.end)
    if len(points) < 2:
        return 36.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lx = max(xs) - min(xs)
    ly = max(ys) - min(ys)
    print(f"Estimated floor plan bounding box: {lx:.3f} m x {ly:.3f} m")
    area = lx * ly
    return max(area, 16.0)


def estimate_structural_self_mass_per_floor(
    input_data: FEMInput,
    story_height: float,
    wall_thickness: float,
    beam_width: float,
    beam_depth: float,
    slab_thickness: float,
    density_kg_m3: float,
    floor_area: Union[float, None] = None,
) -> dict[str, float]:
    area = estimate_floor_area(input_data) if floor_area is None else max(0.0, floor_area)

    wall_length_total = sum(m.length for m in input_data.walls if m.length > 1.0e-9)
    beam_length_total = sum(m.length for m in input_data.beams if m.length > 1.0e-9)

    wall_vol = wall_length_total * max(0.0, wall_thickness) * max(0.0, story_height)
    beam_vol = beam_length_total * max(0.0, beam_width) * max(0.0, beam_depth)
    slab_vol = area * max(0.0, slab_thickness)

    density = max(0.0, density_kg_m3)
    wall_mass = wall_vol * density
    beam_mass = beam_vol * density
    slab_mass = slab_vol * density
    total_mass = wall_mass + beam_mass + slab_mass

    return {
        "wall_mass": wall_mass,
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
