import math
from dataclasses import dataclass
from typing import Dict


@dataclass
class OpenSeesEquivalentAnalyzer:
    """
    Equivalent shear-building analyzer based on OpenSeesPy.

    Notes:
    - This is an initial, high-throughput model for dataset generation.
    - It uses layout summary features (total wall length / plan span) and grouped sizes.
    - Replace this with refined wall-beam explicit model in the next phase.
    """

    constraints: Dict[str, float]

    def evaluate(self, topo_feat: Dict[str, float], design: Dict[str, float], context: Dict) -> Dict[str, object]:
        try:
            import openseespy.opensees as ops
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenSeesPy is required. Install with: pip install openseespy"
            ) from exc

        num_stories = int(context["num_stories"])
        story_h = float(context["story_height_m"])
        dead = float(context["dead_load_kpa"])
        live = float(context["live_load_kpa"])

        span_x = topo_feat["plan_span_x_mm"] / 1000.0
        span_y = topo_feat["plan_span_y_mm"] / 1000.0
        area = max(span_x * span_y, 10.0)
        wall_len = max(topo_feat["total_wall_length_mm"] / 1000.0, 1.0)

        t_wall = design["wall_thickness_mm"] / 1000.0
        fc_mpa = design["concrete_grade"]
        sf = design["group_strength_factor"]

        # Equivalent section parameters for a single stick per story
        e_pa = 3.0e10 * (fc_mpa / 30.0) ** 0.5
        e_kn_m2 = e_pa / 1000.0
        a_eff = max(wall_len * t_wall * sf, 0.05)
        iz_eff = max(wall_len * (t_wall**3) / 12.0 * sf, 1e-5)

        floor_w = (dead + 0.5 * live) * area  # kN
        floor_m = floor_w / 9.81  # kN*s^2/m

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.geomTransf("Linear", 1)

        for i in range(num_stories + 1):
            node_tag = i + 1
            ops.node(node_tag, 0.0, i * story_h)
            if i == 0:
                ops.fix(node_tag, 1, 1, 1)
            else:
                ops.mass(node_tag, floor_m, 1e-8, 1e-8)

        for i in range(1, num_stories + 1):
            ops.element("elasticBeamColumn", i, i, i + 1, a_eff, e_kn_m2, iz_eff, 1)

        # Gravity preload
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for i in range(1, num_stories + 1):
            ops.load(i + 1, 0.0, -floor_w, 0.0)

        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.test("NormDispIncr", 1.0e-8, 20)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        ok_g = ops.analyze(1)
        if ok_g != 0:
            return self._failed_result("gravity_not_converged")

        # Keep gravity effects and add equivalent lateral load
        ops.loadConst("-time", 0.0)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        story_idx_sum = num_stories * (num_stories + 1) / 2.0
        lateral_base = 0.03 * floor_w * num_stories  # coarse seismic coefficient
        for i in range(1, num_stories + 1):
            fi = lateral_base * (i / story_idx_sum)
            ops.load(i + 1, fi, 0.0, 0.0)

        ok_l = ops.analyze(1)
        if ok_l != 0:
            return self._failed_result("lateral_not_converged")

        # Story drift
        drift_max = 0.0
        for i in range(1, num_stories + 1):
            u_top = ops.nodeDisp(i + 1, 1)
            u_bot = ops.nodeDisp(i, 1)
            drift_i = abs((u_top - u_bot) / story_h)
            drift_max = max(drift_max, drift_i)

        # Base shear / shear-weight ratio
        ops.reactions()
        base_shear = abs(ops.nodeReaction(1, 1))
        total_weight = floor_w * num_stories
        shear_weight_ratio = base_shear / max(total_weight, 1e-6)

        # Axial ratio from max element axial force
        axial_max = 0.0
        for ele_tag in range(1, num_stories + 1):
            f = ops.eleForce(ele_tag)
            p_i = abs(float(f[0])) if len(f) > 0 else 0.0
            p_j = abs(float(f[3])) if len(f) > 3 else 0.0
            axial_max = max(axial_max, p_i, p_j)
        fc_kn_m2 = fc_mpa * 1000.0
        axial_ratio = axial_max / max(fc_kn_m2 * a_eff, 1e-6)

        # First-mode period
        lambdas = ops.eigen(1)
        if not lambdas:
            period = 99.0
        else:
            lam = float(lambdas[0])
            period = (2.0 * math.pi / math.sqrt(lam)) if lam > 0 else 99.0

        drift_margin = self.constraints["drift_limit"] - drift_max
        axial_margin = self.constraints["axial_ratio_limit"] - axial_ratio
        shear_margin = shear_weight_ratio - self.constraints["shear_weight_ratio_min"]
        feasible = 1.0 if (drift_margin >= 0 and axial_margin >= 0 and shear_margin >= 0) else 0.0

        return {
            "drift": drift_max,
            "period": period,
            "axial_ratio_max": axial_ratio,
            "shear_weight_ratio": shear_weight_ratio,
            "drift_margin": drift_margin,
            "axial_margin": axial_margin,
            "shear_margin": shear_margin,
            "cost_index": self._cost_index(topo_feat, design, num_stories, area),
            "feasible": feasible,
            "analysis_failed": 0.0,
            "failure_reason": "",
        }

    @staticmethod
    def _cost_index(topo_feat: Dict[str, float], design: Dict[str, float], n_story: int, area: float) -> float:
        wall_len = topo_feat["total_wall_length_mm"] / 1000.0
        beam_len = topo_feat["total_beam_length_mm"] / 1000.0
        t = design["wall_thickness_mm"] / 1000.0
        b = design["beam_width_mm"] / 1000.0
        h = design["beam_depth_mm"] / 1000.0
        concrete_index = (wall_len * t + 0.25 * area * 0.12) * n_story
        steel_index = (0.70 * wall_len * t + 0.30 * beam_len * b * h) * n_story
        return 0.55 * concrete_index + 1.8 * steel_index

    @staticmethod
    def _failed_result(reason: str) -> Dict[str, object]:
        return {
            "drift": 1.0,
            "period": 99.0,
            "axial_ratio_max": 9.9,
            "shear_weight_ratio": 0.0,
            "drift_margin": -1.0,
            "axial_margin": -9.0,
            "shear_margin": -1.0,
            "cost_index": 1e9,
            "feasible": 0.0,
            "analysis_failed": 1.0,
            "failure_reason": reason,
        }
