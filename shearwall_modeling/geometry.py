import random
from pathlib import Path
from typing import Union

from .domain import FEMInput


def choose_scale_factor(input_data: FEMInput, low: float = 6.0, high: float = 12.0, seed: int = 42) -> float:
    lengths = [m.length for m in input_data.all_members() if m.length > 1e-9]
    if not lengths:
        return 1.0
    max_len = max(lengths)
    rng = random.Random(seed)
    target_max = rng.uniform(low, high)
    return target_max / max_len


def estimate_floor_area(input_data: FEMInput) -> float:
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
    area = lx * ly
    return max(area, 16.0)


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
