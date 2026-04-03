import math
from typing import Any


def modal_periods_from_eigenvalues(eigen_values: list[float]) -> list[float]:
    return [2.0 * math.pi / math.sqrt(value) for value in eigen_values if value > 1.0e-12]


def safe_abs_float_list(values: Any, max_len: int) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * max_len

    result: list[float] = []
    for value in values[:max_len]:
        try:
            result.append(abs(float(value)))
        except (TypeError, ValueError):
            result.append(0.0)
    if len(result) < max_len:
        result.extend([0.0] * (max_len - len(result)))
    return result


def identify_dominant_modes(modal_props: dict[str, Any]) -> tuple[int | None, int | None]:
    num_modes = len(modal_props.get("eigenLambda", []))
    if num_modes <= 0:
        return None, None

    mass_x = safe_abs_float_list(modal_props.get("partiMassRatiosMX"), num_modes)
    mass_y = safe_abs_float_list(modal_props.get("partiMassRatiosMY"), num_modes)
    mass_rz = safe_abs_float_list(modal_props.get("partiMassRatiosRMZ"), num_modes)

    translational_mode_index = None
    torsional_mode_index = None
    for index in range(num_modes):
        translational_ratio = max(mass_x[index], mass_y[index])
        torsional_ratio = mass_rz[index]
        if translational_mode_index is None and translational_ratio >= torsional_ratio and translational_ratio > 0.0:
            translational_mode_index = index + 1
        if torsional_mode_index is None and torsional_ratio > translational_ratio and torsional_ratio > 0.0:
            torsional_mode_index = index + 1
        if translational_mode_index is not None and torsional_mode_index is not None:
            break
    return translational_mode_index, torsional_mode_index

