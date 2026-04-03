from dataclasses import dataclass

CONCRETE_ELASTIC_MODULUS_MPA: dict[str, float] = {
    "C30": 3.00e4,
    "C35": 3.15e4,
    "C40": 3.25e4,
    "C45": 3.35e4,
    "C50": 3.45e4,
}

CONCRETE_COMPRESSIVE_STRENGTH_PA: dict[str, float] = {
    "C30": 14.3e6,
    "C35": 16.7e6,
    "C40": 19.1e6,
    "C45": 21.2e6,
    "C50": 23.1e6,
}

SEISMIC_ALPHA_MAX_BY_INTENSITY: dict[float, float] = {
    6.0: 0.04,
    7.0: 0.08,
    7.5: 0.12,
    8.0: 0.16,
    8.5: 0.24,
    9.0: 0.32,
}

GB50011_TG_BY_SITE_CLASS: dict[str, dict[int, float]] = {
    "I0": {1: 0.20, 2: 0.25, 3: 0.30},
    "I": {1: 0.25, 2: 0.30, 3: 0.35},
    "II": {1: 0.35, 2: 0.40, 3: 0.45},
    "III": {1: 0.45, 2: 0.55, 3: 0.65},
    "IV": {1: 0.65, 2: 0.75, 3: 0.90},
}

MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY: dict[float, float] = {
    6.0: 0.008,
    7.0: 0.016,
    7.5: 0.024,
    8.0: 0.032,
    8.5: 0.048,
    9.0: 0.064,
}


@dataclass(frozen=True)
class RectSlabFilterBounds:
    min_span: float = 2.0
    max_span: float = 8.0
    min_depth: float = 3.0
    max_depth: float = 12.0
    max_aspect_ratio: float = 3.5
    median_span_factor: float = 0.55
    median_depth_factor: float = 0.55
    median_area_factor: float = 0.35
    min_span_floor: float = 0.8
    min_depth_floor: float = 1.0
    min_area_floor: float = 2.0

