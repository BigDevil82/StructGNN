from dataclasses import dataclass
from typing import Dict


@dataclass
class ProxyAnalyzer:
    """Fast deterministic proxy for early-stage dataset bootstrapping."""

    constraints: Dict[str, float]

    def evaluate(self, topo_feat: Dict[str, float], design: Dict[str, float], context: Dict) -> Dict[str, float]:
        num_stories = float(context["num_stories"])
        story_height = float(context["story_height_m"])
        h_total = max(num_stories * story_height, 1.0)
        dead = float(context["dead_load_kpa"])
        live = float(context["live_load_kpa"])

        span_x = topo_feat["plan_span_x_mm"] / 1000.0
        span_y = topo_feat["plan_span_y_mm"] / 1000.0
        area = max(span_x * span_y, 10.0)
        wall_len = topo_feat["total_wall_length_mm"] / 1000.0

        t_wall = design["wall_thickness_mm"] / 1000.0
        beam_b = design["beam_width_mm"] / 1000.0
        beam_h = design["beam_depth_mm"] / 1000.0
        fc = design["concrete_grade"]
        sf = design["group_strength_factor"]

        stiffness = max(wall_len * t_wall * fc * sf, 1e-6)
        lateral_demand = (dead + 0.5 * live) * area * h_total
        drift = 0.0002 + 0.0012 * lateral_demand / (stiffness * 1000.0)

        t_period = 0.08 * (h_total ** 0.75) * (1.0 + 0.20 * (0.30 - t_wall))
        t_period = max(0.2, t_period)

        axial_ratio = min(
            1.2,
            0.10 + 0.35 * (dead + 0.3 * live) * area / (8.0 * wall_len * t_wall * fc * sf + 1e-6),
        )
        shear_weight_ratio = max(0.001, 0.28 * wall_len * t_wall / area)

        steel_index = (
            0.65 * wall_len * t_wall + 0.35 * topo_feat["total_beam_length_mm"] / 1000.0 * beam_b * beam_h
        ) * num_stories
        concrete_index = (wall_len * t_wall + 0.25 * area * 0.12) * num_stories
        cost_index = 0.55 * concrete_index + 1.8 * steel_index

        drift_margin = self.constraints["drift_limit"] - drift
        axial_margin = self.constraints["axial_ratio_limit"] - axial_ratio
        shear_margin = shear_weight_ratio - self.constraints["shear_weight_ratio_min"]
        feasible = 1.0 if (drift_margin >= 0 and axial_margin >= 0 and shear_margin >= 0) else 0.0

        return {
            "drift": drift,
            "period": t_period,
            "axial_ratio_max": axial_ratio,
            "shear_weight_ratio": shear_weight_ratio,
            "drift_margin": drift_margin,
            "axial_margin": axial_margin,
            "shear_margin": shear_margin,
            "cost_index": cost_index,
            "feasible": feasible,
        }
