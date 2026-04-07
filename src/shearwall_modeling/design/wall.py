from ..core.config import MaterialConfig
from .constants import CONCRETE_COMPRESSIVE_STRENGTH_MPA, CONCRETE_TENSILE_STRENGTH_MPA, ReinforcementDesignConstants
from .results import MaterialUsage, WallDesignDemand, WallReinforcementResult


def _resolve_fc_ft(material: MaterialConfig) -> tuple[float, float]:
    grade = material.concrete_grade.strip().upper()
    return CONCRETE_COMPRESSIVE_STRENGTH_MPA[grade], CONCRETE_TENSILE_STRENGTH_MPA[grade]


class WallReinforcementDesigner:
    def __init__(self, constants: ReinforcementDesignConstants | None = None):
        self.constants = constants or ReinforcementDesignConstants()

    def design(self, demand: WallDesignDemand, material: MaterialConfig) -> WallReinforcementResult:
        fc_mpa, ft_mpa = _resolve_fc_ft(material)
        bw_mm = demand.thickness_m * 1000.0
        lw_mm = demand.length_m * 1000.0
        hc_mm = max(2.0 * bw_mm, 0.15 * lw_mm)
        bc_mm = bw_mm
        h0_mm = lw_mm - self.constants.cover_to_centroid_mm

        vertical_area = self.constants.wall_vertical_min_ratio * bw_mm * 1000.0
        horizontal_area = self._design_horizontal_distributed_steel(demand, bw_mm, h0_mm, ft_mpa)
        boundary_type = self._resolve_boundary_type(demand.axial_ratio, demand.is_bottom_reinforced_zone)
        boundary_longitudinal, boundary_stirrup_area, boundary_spacing, over_reinforced, messages = (
            self._design_boundary_steel(demand, bc_mm, hc_mm, h0_mm, fc_mpa, boundary_type)
        )

        concrete_kg = demand.thickness_m * demand.length_m * demand.story_height_m * self.constants.concrete_density_kg_m3
        vertical_kg = (
            vertical_area * demand.length_m * demand.story_height_m * self.constants.steel_density_kg_m3 * 1.0e-6
        )
        horizontal_kg = (
            horizontal_area * demand.length_m * demand.story_height_m * self.constants.steel_density_kg_m3 * 1.0e-6
        )
        boundary_kg = (
            (
                boundary_longitudinal * hc_mm / 1000.0
                + boundary_stirrup_area * (hc_mm / max(boundary_spacing, 1.0)) * bc_mm / 1000.0
            )
            * 2.0
            * self.constants.steel_density_kg_m3
            * 1.0e-6
        )

        return WallReinforcementResult(
            wall_id=demand.wall_id,
            story=demand.story,
            vertical_distributed_steel_mm2_per_m=vertical_area,
            horizontal_distributed_steel_mm2_per_m=horizontal_area,
            boundary_longitudinal_steel_mm2=boundary_longitudinal,
            boundary_stirrup_area_mm2=boundary_stirrup_area,
            boundary_stirrup_spacing_mm=boundary_spacing,
            boundary_type=boundary_type,
            material_usage=MaterialUsage(
                concrete_kg=concrete_kg,
                steel_kg=vertical_kg + horizontal_kg + boundary_kg,
            ),
            is_over_reinforced=over_reinforced,
            is_section_insufficient=False,
            messages=messages,
        )

    def _design_horizontal_distributed_steel(
        self, demand: WallDesignDemand, wall_thickness_mm: float, effective_length_mm: float, ft_mpa: float
    ) -> float:
        spacing_mm = self.constants.wall_horizontal_spacing_mm
        design_shear_n = self.constants.gamma_re_shear * abs(demand.shear_n)
        steel_area = (
            (design_shear_n - 0.4 * ft_mpa * wall_thickness_mm * effective_length_mm)
            * spacing_mm
            / max(0.8 * self.constants.fyv_mpa * effective_length_mm, 1.0e-9)
        )
        steel_area_per_m = max(
            steel_area / max(spacing_mm, 1.0e-9) * 1000.0,
            self.constants.wall_horizontal_min_ratio * wall_thickness_mm * 1000.0,
        )
        return steel_area_per_m

    def _resolve_boundary_type(self, axial_ratio: float, is_bottom_reinforced_zone: bool) -> str:
        threshold = 0.3 if is_bottom_reinforced_zone else 0.4
        return "constrained" if axial_ratio > threshold else "constructive"

    def _design_boundary_steel(
        self,
        demand: WallDesignDemand,
        boundary_width_mm: float,
        boundary_height_mm: float,
        effective_length_mm: float,
        fc_mpa: float,
        boundary_type: str,
    ) -> tuple[float, float, float, bool, list[str]]:
        messages: list[str] = []
        design_axial_n = self.constants.gamma_re_bending * abs(demand.axial_force_n)
        design_moment_n_mm = self.constants.gamma_re_bending * abs(demand.moment_n_m) * 1000.0
        x_mm = design_axial_n / max(self.constants.alpha_1 * fc_mpa * boundary_width_mm, 1.0e-9)
        xi = x_mm / max(effective_length_mm, 1.0e-9)
        over_reinforced = xi > self.constants.xi_b
        if over_reinforced:
            messages.append("boundary_over_reinforced")

        numerator = design_moment_n_mm - self.constants.alpha_1 * fc_mpa * boundary_width_mm * x_mm * (
            effective_length_mm - 0.5 * x_mm
        )
        longitudinal_area = max(
            numerator / max(self.constants.fy_mpa * (effective_length_mm - self.constants.cover_to_centroid_mm), 1.0e-9),
            0.0,
        )
        min_ratio = (
            self.constants.boundary_constrained_min_ratio
            if boundary_type == "constrained"
            else self.constants.boundary_constructive_min_ratio
        )
        longitudinal_area = max(longitudinal_area, min_ratio * boundary_width_mm * boundary_height_mm)
        if longitudinal_area == min_ratio * boundary_width_mm * boundary_height_mm:
            messages.append("minimum_boundary_longitudinal_steel_governs")

        volume_ratio = 0.006 if boundary_type == "constrained" else 0.004
        spacing_mm = (
            self.constants.boundary_stirrup_spacing_constrained_mm
            if boundary_type == "constrained"
            else self.constants.boundary_stirrup_spacing_constructive_mm
        )
        stirrup_area = (
            volume_ratio
            * boundary_width_mm
            * boundary_height_mm
            * spacing_mm
            / max(self.constants.boundary_stirrup_limb_spacing_mm, 1.0e-9)
        )
        return longitudinal_area, stirrup_area, spacing_mm, over_reinforced, messages

