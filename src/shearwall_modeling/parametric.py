import math
import random
from dataclasses import dataclass
from typing import Any

from .config import MaterialConfig, ModelConfig, SectionConfig, SeismicConfig, StandardStoryGroupConfig


@dataclass(frozen=True)
class ParametricModelParams:
    N: int
    t_w_bot: int
    h_b: int
    b_b: int
    h_s: int
    conc_bot: str
    intensity: float
    site_class: str
    seismic_group: int


PARAM_SPACE: dict[str, list[Any]] = {
    "N": list(range(18, 34)),
    "t_w_bot": [200, 250, 300],
    "h_b": [400, 450, 500, 550, 600],
    "b_b": [200, 250, 300],
    "h_s": [100, 120, 150],
    "conc_bot": ["C30", "C35", "C40", "C45", "C50"],
    "intensity": [6.0, 7.0, 7.5, 8.0, 8.5, 9.0],
    "site_class": ["I0", "I", "II", "III", "IV"],
    "seismic_group": [1, 2, 3],
}


def sample_parametric_model_params(rng: random.Random | None = None) -> ParametricModelParams:
    sampler = rng or random.Random()
    return ParametricModelParams(
        N=sampler.choice(PARAM_SPACE["N"]),
        t_w_bot=sampler.choice(PARAM_SPACE["t_w_bot"]),
        h_b=sampler.choice(PARAM_SPACE["h_b"]),
        b_b=sampler.choice(PARAM_SPACE["b_b"]),
        h_s=sampler.choice(PARAM_SPACE["h_s"]),
        conc_bot=sampler.choice(PARAM_SPACE["conc_bot"]),
        intensity=sampler.choice(PARAM_SPACE["intensity"]),
        site_class=sampler.choice(PARAM_SPACE["site_class"]),
        seismic_group=sampler.choice(PARAM_SPACE["seismic_group"]),
    )


def build_model_config_from_params(
    params: ParametricModelParams | dict[str, Any],
    *,
    story_height: float = 2.9,
    num_modes: int = 6,
) -> ModelConfig:
    parsed = _parse_params(params)

    n_bottom, n_middle, n_top = _split_stories(parsed.N)

    t_w_mid = max(160, parsed.t_w_bot - 20)
    t_w_top = max(160, parsed.t_w_bot - 40)

    conc_mid = _downgrade_concrete_grade(parsed.conc_bot, steps=1)
    conc_top = _downgrade_concrete_grade(parsed.conc_bot, steps=2)

    beam_width = parsed.b_b / 1000.0
    beam_depth = parsed.h_b / 1000.0
    slab_thickness = parsed.h_s / 1000.0

    def make_section(wall_thickness_mm: int) -> SectionConfig:
        return SectionConfig(
            wall_thickness=wall_thickness_mm / 1000.0,
            beam_width=beam_width,
            beam_depth=beam_depth,
            secondary_beam_width=beam_width,
            secondary_beam_depth=beam_depth,
            slab_thickness=slab_thickness,
        )

    groups = [
        StandardStoryGroupConfig(
            count=n_bottom,
            story_height=story_height,
            section=make_section(parsed.t_w_bot),
            material=MaterialConfig(concrete_grade=parsed.conc_bot),
        ),
        StandardStoryGroupConfig(
            count=n_middle,
            story_height=story_height,
            section=make_section(t_w_mid),
            material=MaterialConfig(concrete_grade=conc_mid),
        ),
        StandardStoryGroupConfig(
            count=n_top,
            story_height=story_height,
            section=make_section(t_w_top),
            material=MaterialConfig(concrete_grade=conc_top),
        ),
    ]

    seismic = SeismicConfig(
        intensity=parsed.intensity,
        site_class=parsed.site_class,
        seismic_group=parsed.seismic_group,
    )

    return ModelConfig(
        standard_story_groups=groups,
        seismic=seismic,
        num_modes=num_modes,
    )


def _split_stories(total_stories: int) -> tuple[int, int, int]:
    if total_stories < 3:
        raise ValueError("N must be at least 3 to create bottom/middle/top segments.")

    n1 = min(10, max(2, int(math.floor(0.3 * total_stories))))
    n2 = max(n1 + 1, int(math.floor(0.7 * total_stories)))
    n2 = min(total_stories - 1, n2)

    n_bottom = n1
    n_middle = n2 - n1
    n_top = total_stories - n2

    if n_middle <= 0 or n_top <= 0:
        raise ValueError(f"Invalid story split for N={total_stories}: ({n_bottom}, {n_middle}, {n_top})")

    return n_bottom, n_middle, n_top


def _downgrade_concrete_grade(grade: str, steps: int) -> str:
    order = ["C30", "C35", "C40", "C45", "C50"]
    normalized = grade.strip().upper()
    if not normalized.startswith("C"):
        normalized = f"C{normalized}"
    if normalized not in order:
        supported = ", ".join(order)
        raise ValueError(f"Unsupported concrete grade={grade}. Supported grades: {supported}")

    idx = order.index(normalized)
    downgraded_idx = max(0, idx - steps)
    return order[downgraded_idx]


def _parse_params(params: ParametricModelParams | dict[str, Any]) -> ParametricModelParams:
    if isinstance(params, ParametricModelParams):
        return params

    required = [
        "N",
        "t_w_bot",
        "h_b",
        "b_b",
        "h_s",
        "conc_bot",
        "intensity",
        "site_class",
        "seismic_group",
    ]
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"Missing required parameters: {', '.join(missing)}")

    parsed = ParametricModelParams(
        N=int(params["N"]),
        t_w_bot=int(params["t_w_bot"]),
        h_b=int(params["h_b"]),
        b_b=int(params["b_b"]),
        h_s=int(params["h_s"]),
        conc_bot=str(params["conc_bot"]),
        intensity=float(params["intensity"]),
        site_class=str(params["site_class"]),
        seismic_group=int(params["seismic_group"]),
    )

    _validate_sampled_params(parsed)
    return parsed


def _validate_sampled_params(params: ParametricModelParams) -> None:
    for name in PARAM_SPACE:
        value = getattr(params, name)
        if value not in PARAM_SPACE[name]:
            raise ValueError(f"{name}={value} is outside the predefined parameter space.")
