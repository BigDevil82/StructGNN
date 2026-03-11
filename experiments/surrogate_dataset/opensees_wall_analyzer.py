import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class OpenSeesWallElasticAnalyzer:
    constraints: Dict[str, float]

    def evaluate(self, topology: Dict, design: Dict[str, float], context: Dict) -> Dict[str, object]:
        try:
            import openseespy.opensees as ops
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenSeesPy is required. Install with: pip install openseespy") from exc

        num_stories = int(context["num_stories"])
        story_h = float(context["story_height_m"])
        dead = float(context["dead_load_kpa"])
        live = float(context["live_load_kpa"])

        span_x = max(topology["summary"].plan_span_x_mm / 1000.0, 1.0)
        span_y = max(topology["summary"].plan_span_y_mm / 1000.0, 1.0)
        floor_area = max(span_x * span_y, 10.0)
        floor_w = (dead + 0.5 * live) * floor_area
        floor_m = floor_w / 9.81

        wall_t = design["wall_thickness_mm"] / 1000.0
        fc_mpa = design["concrete_grade"]
        e_pa = 3.0e10 * (fc_mpa / 30.0) ** 0.5
        e_kn_m2 = e_pa / 1000.0

        walls = topology["shearwalls"]
        x_walls, y_walls = self._split_walls(walls)
        if not x_walls and not y_walls:
            return self._failed_result("no_walls")

        drift_x, period_x, base_shear_x = self._analyze_direction(
            ops=ops,
            wall_count=len(y_walls),  # walls parallel Y resist X drift
            wall_lengths=y_walls,
            wall_t=wall_t,
            e_kn_m2=e_kn_m2,
            num_stories=num_stories,
            story_h=story_h,
            floor_w=floor_w,
            floor_m=floor_m,
        )
        drift_y, period_y, base_shear_y = self._analyze_direction(
            ops=ops,
            wall_count=len(x_walls),  # walls parallel X resist Y drift
            wall_lengths=x_walls,
            wall_t=wall_t,
            e_kn_m2=e_kn_m2,
            num_stories=num_stories,
            story_h=story_h,
            floor_w=floor_w,
            floor_m=floor_m,
        )

        if drift_x is None and drift_y is None:
            return self._failed_result("analysis_not_converged")

        drift_vals = [v for v in [drift_x, drift_y] if v is not None]
        period_vals = [v for v in [period_x, period_y] if v is not None]
        shear_vals = [v for v in [base_shear_x, base_shear_y] if v is not None]

        drift_max = max(drift_vals) if drift_vals else 1.0
        period = max(period_vals) if period_vals else 99.0
        base_shear = max(shear_vals) if shear_vals else 0.0
        total_weight = floor_w * num_stories
        shear_weight_ratio = base_shear / max(total_weight, 1e-6)

        wall_area_total = wall_t * sum(x_walls + y_walls)
        fc_kn_m2 = fc_mpa * 1000.0
        gravity_axial = total_weight / max(len(walls), 1)
        axial_ratio = gravity_axial / max(fc_kn_m2 * (wall_area_total / max(len(walls), 1)), 1e-6)

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
            "cost_index": self._cost_index(topology["summary"].to_feature_dict(), design, num_stories, floor_area),
            "feasible": feasible,
            "analysis_failed": 0.0,
            "failure_reason": "",
            "drift_x": drift_x if drift_x is not None else -1.0,
            "drift_y": drift_y if drift_y is not None else -1.0,
            "period_x": period_x if period_x is not None else -1.0,
            "period_y": period_y if period_y is not None else -1.0,
        }

    def _analyze_direction(
        self,
        ops,
        wall_count: int,
        wall_lengths: List[float],
        wall_t: float,
        e_kn_m2: float,
        num_stories: int,
        story_h: float,
        floor_w: float,
        floor_m: float,
    ) -> Tuple[float, float, float]:
        if wall_count <= 0:
            return None, None, None

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.geomTransf("Linear", 1)

        for i in range(num_stories + 1):
            ops.node(i + 1, 0.0, i * story_h)
            if i == 0:
                ops.fix(i + 1, 1, 1, 1)
            else:
                ops.mass(i + 1, floor_m, 1e-9, 1e-9)

        # Multiple walls combined in parallel: sum(A), sum(I)
        area_sum = 0.0
        inertia_sum = 0.0
        for lw in wall_lengths:
            l = max(lw, 0.2)
            area_sum += wall_t * l
            inertia_sum += wall_t * (l**3) / 12.0
        area_sum = max(area_sum, 0.05)
        inertia_sum = max(inertia_sum, 1e-5)

        for i in range(1, num_stories + 1):
            ops.element("elasticBeamColumn", i, i, i + 1, area_sum, e_kn_m2, inertia_sum, 1)

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for i in range(1, num_stories + 1):
            ops.load(i + 1, 0.0, -floor_w, 0.0)

        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.test("NormDispIncr", 1.0e-10, 50)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            return None, None, None

        ops.loadConst("-time", 0.0)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        lateral_base = 0.08 * floor_w * num_stories
        story_idx_sum = num_stories * (num_stories + 1) / 2.0
        for i in range(1, num_stories + 1):
            fi = lateral_base * (i / story_idx_sum)
            ops.load(i + 1, fi, 0.0, 0.0)

        if ops.analyze(1) != 0:
            return None, None, None

        drift_max = 0.0
        for i in range(1, num_stories + 1):
            d = abs((ops.nodeDisp(i + 1, 1) - ops.nodeDisp(i, 1)) / story_h)
            drift_max = max(drift_max, d)

        ops.reactions()
        base_shear = abs(ops.nodeReaction(1, 1))

        lambdas = ops.eigen(1)
        if not lambdas:
            period = 99.0
        else:
            lam = float(lambdas[0])
            period = (2.0 * math.pi / math.sqrt(lam)) if lam > 0 else 99.0

        return drift_max, period, base_shear

    @staticmethod
    def _split_walls(walls: List[Dict]) -> Tuple[List[float], List[float]]:
        x_walls: List[float] = []
        y_walls: List[float] = []
        tol = 1.0e-6
        for w in walls:
            s = w.get("start", [0.0, 0.0])
            e = w.get("end", [0.0, 0.0])
            dx = float(e[0]) - float(s[0])
            dy = float(e[1]) - float(s[1])
            length_m = float(w.get("length", 0.0)) / 1000.0
            if length_m <= 0.0:
                continue
            if abs(dy) <= tol and abs(dx) > tol:
                x_walls.append(length_m)
            elif abs(dx) <= tol and abs(dy) > tol:
                y_walls.append(length_m)
        return x_walls, y_walls

    @staticmethod
    def _cost_index(topo_feat: Dict[str, float], design: Dict[str, float], n_story: int, area: float) -> float:
        wall_len = topo_feat["total_wall_length_mm"] / 1000.0
        beam_len = topo_feat["total_beam_length_mm"] / 1000.0
        t = design["wall_thickness_mm"] / 1000.0
        b = design["beam_width_mm"] / 1000.0
        h = design["beam_depth_mm"] / 1000.0
        concrete_index = (wall_len * t + 0.25 * area * 0.12) * n_story
        steel_index = (0.65 * wall_len * t + 0.35 * beam_len * b * h) * n_story
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
            "drift_x": -1.0,
            "drift_y": -1.0,
            "period_x": -1.0,
            "period_y": -1.0,
        }

