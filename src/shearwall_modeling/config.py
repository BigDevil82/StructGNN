from dataclasses import dataclass, field
from typing import Optional


def _gb50011_spectrum_shape_params(damping_ratio: float) -> tuple[float, float, float]:
    xi = damping_ratio
    gamma = 0.9 + (0.05 - xi) / (0.3 + 6.0 * xi)
    eta1 = 0.02 + (0.05 - xi) / (4.0 + 32.0 * xi)
    eta2 = 1.0 + (0.05 - xi) / (0.08 + 1.6 * xi)

    eta1 = max(0.0, eta1)
    eta2 = min(1.0, max(0.55, eta2))
    return gamma, eta1, eta2


def _gb50011_alpha(T: float, alpha_max: float, Tg: float, damping_ratio: float) -> float:
    gamma, eta1, eta2 = _gb50011_spectrum_shape_params(damping_ratio)

    if T <= 0.1:
        return alpha_max * (0.45 + (eta2 - 0.45) * (T / 0.1))
    if T <= Tg:
        return eta2 * alpha_max
    if T <= 5.0 * Tg:
        return eta2 * alpha_max * ((Tg / T) ** gamma)
    if T <= 6.0:
        return alpha_max * (eta2 * (0.2**gamma) - eta1 * (T - 5.0 * Tg))
    return alpha_max * max(0.0, eta2 * (0.2**gamma) - eta1 * (6.0 - 5.0 * Tg))


def _resolve_gb50011_design_params(
    intensity: float,
    site_class: str,
    seismic_group: int,
    alpha_max_override: Optional[float],
    Tg_override: Optional[float],
) -> tuple[float, float]:
    alpha_max_map = {
        6.0: 0.04,
        7.0: 0.08,
        7.5: 0.12,
        8.0: 0.16,
        8.5: 0.24,
        9.0: 0.32,
    }

    tg_map = {
        "I0": {1: 0.20, 2: 0.25, 3: 0.30},
        "I": {1: 0.25, 2: 0.30, 3: 0.35},
        "II": {1: 0.35, 2: 0.40, 3: 0.45},
        "III": {1: 0.45, 2: 0.55, 3: 0.65},
        "IV": {1: 0.65, 2: 0.75, 3: 0.90},
    }

    acc_key = round(float(intensity), 2)
    site_key = str(site_class).upper()
    group_key = int(seismic_group)

    if alpha_max_override is None:
        if acc_key not in alpha_max_map:
            raise ValueError(f"Unsupported intensity={intensity}. " "Set alpha_max_override to continue.")
        alpha_max = alpha_max_map[acc_key]
    else:
        alpha_max = float(alpha_max_override)

    if Tg_override is None:
        if site_key not in tg_map or group_key not in tg_map[site_key]:
            raise ValueError(
                f"Unsupported site/group combination: site_class={site_class}, "
                f"seismic_group={seismic_group}. Set Tg_override to continue."
            )
        tg = tg_map[site_key][group_key]
    else:
        tg = float(Tg_override)

    return alpha_max, tg


def _build_gb50011_spectrum(
    alpha_max: float,
    Tg: float,
    damping_ratio: float,
    gravity: float,
    t_max: float,
    dt: float,
) -> tuple[list[float], list[float]]:
    if dt <= 0.0:
        raise ValueError("spectrum_dt must be greater than 0.")
    if t_max <= 0.0:
        raise ValueError("spectrum_t_max must be greater than 0.")

    periods: list[float] = []
    spectral_acc: list[float] = []
    steps = int(t_max / dt)
    for i in range(steps + 1):
        t = i * dt
        alpha = _gb50011_alpha(t, alpha_max, Tg, damping_ratio=damping_ratio)
        periods.append(t)
        spectral_acc.append(alpha * gravity)

    return periods, spectral_acc


@dataclass
class MaterialConfig:
    E: float = 3.0e10
    G: float = 1.2e10
    density_kg_m3: float = 2550.0


@dataclass
class SectionConfig:
    wall_thickness: float = 0.2
    beam_width: float = 0.3
    beam_depth: float = 0.5
    slab_thickness: float = 0.12


@dataclass
class SeismicConfig:
    periods: list[float] = field(default_factory=list)
    sa: list[float] = field(default_factory=list)
    damping_ratio: float = 0.05
    combination_method: str = "CQC"
    design_code: str = "GB50011"
    intensity: int = 7
    site_class: str = "II"
    seismic_group: int = 1
    alpha_max_override: Optional[float] = None
    Tg_override: Optional[float] = None
    spectrum_t_max: float = 6.0
    spectrum_dt: float = 0.01
    gravity: float = 9.81

    def __post_init__(self) -> None:
        if self.periods and self.sa:
            return
        if self.periods or self.sa:
            raise ValueError("seismic.periods and seismic.sa should be both provided or both omitted.")

        if self.design_code.upper() != "GB50011":
            raise ValueError(
                f"Unsupported design_code={self.design_code}. " "Currently only GB50011 is implemented."
            )

        alpha_max, tg = _resolve_gb50011_design_params(
            intensity=self.intensity,
            site_class=self.site_class,
            seismic_group=self.seismic_group,
            alpha_max_override=self.alpha_max_override,
            Tg_override=self.Tg_override,
        )

        self.periods, self.sa = _build_gb50011_spectrum(
            alpha_max=alpha_max,
            Tg=tg,
            damping_ratio=self.damping_ratio,
            gravity=self.gravity,
            t_max=self.spectrum_t_max,
            dt=self.spectrum_dt,
        )


@dataclass
class MassSourceConfig:
    dead_kpa: float = 5.0
    live_kpa: float = 2.0
    dead_factor: float = 1.0
    live_factor: float = 0.5
    include_structural_self_weight: bool = True
    gravity: float = 9.81

    def load_to_mass_per_area(self) -> float:
        # Convert ETABS-like source loads (kN/m^2) to kg/m^2.
        source_kpa = self.dead_factor * self.dead_kpa + self.live_factor * self.live_kpa
        return source_kpa * 1000.0 / self.gravity


@dataclass
class ModelConfig:
    standard_story_groups: list["StandardStoryGroupConfig"]
    mass_per_area: float = 1000.0
    mass_source: MassSourceConfig = field(default_factory=MassSourceConfig)
    material: MaterialConfig = field(default_factory=MaterialConfig)
    section: SectionConfig = field(default_factory=SectionConfig)
    seismic: SeismicConfig = field(default_factory=SeismicConfig)
    num_modes: int = 6

    def __post_init__(self) -> None:
        if not self.standard_story_groups:
            raise ValueError("standard_story_groups is required and cannot be empty.")
        for group in self.standard_story_groups:
            if group.count <= 0:
                raise ValueError("standard_story_groups.count must be greater than 0.")
            if group.story_height <= 0.0:
                raise ValueError("standard_story_groups.story_height must be greater than 0.")

    @property
    def num_stories(self) -> int:
        return sum(group.count for group in self.standard_story_groups)

    def resolve_story_profiles(self) -> list["StoryProfile"]:
        profiles: list[StoryProfile] = []

        z_bottom = 0.0
        story_idx = 1
        for group in self.standard_story_groups:
            section = group.section or self.section
            mass_source = group.mass_source or self.mass_source
            story_height = group.story_height

            for _ in range(group.count):
                z_top = z_bottom + story_height
                profiles.append(
                    StoryProfile(
                        story=story_idx,
                        story_height=story_height,
                        z_bottom=z_bottom,
                        z_top=z_top,
                        section=section,
                        mass_source=mass_source,
                    )
                )
                z_bottom = z_top
                story_idx += 1

        return profiles

    def get_story_heights(self) -> list[float]:
        return [profile.story_height for profile in self.resolve_story_profiles()]


@dataclass
class StandardStoryGroupConfig:
    count: int
    story_height: float
    section: Optional[SectionConfig] = None
    mass_source: Optional[MassSourceConfig] = None


@dataclass
class StoryProfile:
    story: int
    story_height: float
    z_bottom: float
    z_top: float
    section: SectionConfig
    mass_source: MassSourceConfig
