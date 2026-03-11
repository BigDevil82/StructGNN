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

        slab_t = float(context.get("slab_thickness_mm", 120.0)) / 1000.0
        density = float(context.get("density_kg_m3", 2500.0))
        live_mass_factor = float(context.get("live_mass_factor", 0.5))
        self_weight_scale = float(context.get("self_weight_scale", 0.65))
        seismic_coeff = float(context.get("seismic_coeff", 0.02))
        stiffness_reduction = float(context.get("stiffness_reduction", 0.002))
        beam_stiffness_factor = float(context.get("beam_stiffness_factor", 0.10))
        auto_calibrate = bool(context.get("auto_calibrate", False))
        target_period = float(context.get("target_period_s", 0.0))
        target_drift = float(context.get("target_drift_ratio", 0.0))
        target_mass_dead = float(context.get("target_mass_dead_t", 0.0))
        target_mass_live = float(context.get("target_mass_live_t", 0.0))

        wall_t = float(design["wall_thickness_mm"]) / 1000.0
        beam_b = float(design["beam_width_mm"]) / 1000.0
        beam_h = float(design["beam_depth_mm"]) / 1000.0
        fc = float(design["concrete_grade"])
        e = 4700.0 * math.sqrt(fc) * 1000.0  # kN/m^2

        span_x = max(topology["summary"].plan_span_x_mm / 1000.0, 1.0)
        span_y = max(topology["summary"].plan_span_y_mm / 1000.0, 1.0)
        floor_area = max(span_x * span_y, 10.0)

        walls_x, walls_y = self._split_members(topology["shearwalls"])
        beams_x, beams_y = self._split_members(topology["beams"])
        if not walls_x and not walls_y:
            return self._failed_result("no_walls")

        mass_dead, mass_live, mass_total = self._estimate_mass_ton(
            floor_area=floor_area,
            total_wall_len=sum(walls_x + walls_y),
            total_beam_len=sum(beams_x + beams_y),
            wall_t=wall_t,
            beam_b=beam_b,
            beam_h=beam_h,
            slab_t=slab_t,
            density=density,
            dead=dead,
            live=live,
            live_mass_factor=live_mass_factor,
            self_weight_scale=self_weight_scale,
            story_h=story_h,
            num_stories=num_stories,
        )
        mass_calib_dead = 1.0
        mass_calib_live = 1.0
        if target_mass_dead > 0.0 and mass_dead > 0.0:
            mass_calib_dead = target_mass_dead / mass_dead
            self_weight_scale *= mass_calib_dead
        if target_mass_live > 0.0 and mass_live > 0.0:
            mass_calib_live = target_mass_live / mass_live
            live_mass_factor *= mass_calib_live

        if mass_calib_dead != 1.0 or mass_calib_live != 1.0:
            mass_dead, mass_live, mass_total = self._estimate_mass_ton(
                floor_area=floor_area,
                total_wall_len=sum(walls_x + walls_y),
                total_beam_len=sum(beams_x + beams_y),
                wall_t=wall_t,
                beam_b=beam_b,
                beam_h=beam_h,
                slab_t=slab_t,
                density=density,
                dead=dead,
                live=live,
                live_mass_factor=live_mass_factor,
                self_weight_scale=self_weight_scale,
                story_h=story_h,
                num_stories=num_stories,
            )
        if mass_total <= 0.0:
            return self._failed_result("invalid_mass")

        floor_mass = mass_total / num_stories

        drift_x, period_x = self._analyze_direction(
            ops=ops,
            wall_lengths=walls_y,
            beam_lengths=beams_x,
            wall_t=wall_t,
            beam_b=beam_b,
            beam_h=beam_h,
            e=e,
            story_h=story_h,
            num_stories=num_stories,
            floor_mass=floor_mass,
            seismic_coeff=seismic_coeff,
            stiffness_reduction=stiffness_reduction,
            beam_stiffness_factor=beam_stiffness_factor,
        )
        drift_y, period_y = self._analyze_direction(
            ops=ops,
            wall_lengths=walls_x,
            beam_lengths=beams_y,
            wall_t=wall_t,
            beam_b=beam_b,
            beam_h=beam_h,
            e=e,
            story_h=story_h,
            num_stories=num_stories,
            floor_mass=floor_mass,
            seismic_coeff=seismic_coeff,
            stiffness_reduction=stiffness_reduction,
            beam_stiffness_factor=beam_stiffness_factor,
        )
        if drift_x is None and drift_y is None:
            return self._failed_result("analysis_not_converged")

        drift = max(v for v in [drift_x, drift_y] if v is not None)
        period = max(v for v in [period_x, period_y] if v is not None)
        calib_k = 1.0
        calib_s = 1.0

        if auto_calibrate and (target_period > 0.0 or target_drift > 0.0):
            if target_period > 0.0 and period > 0.0:
                calib_k = (period / target_period) ** 2
                stiffness_reduction = max(stiffness_reduction * calib_k, 1.0e-6)

            if target_drift > 0.0 and drift > 0.0:
                drift_after_k = drift / max(calib_k, 1.0e-6)
                calib_s = target_drift / max(drift_after_k, 1.0e-12)
                seismic_coeff = max(seismic_coeff * calib_s, 1.0e-6)

            drift_x, period_x = self._analyze_direction(
                ops=ops,
                wall_lengths=walls_y,
                beam_lengths=beams_x,
                wall_t=wall_t,
                beam_b=beam_b,
                beam_h=beam_h,
                e=e,
                story_h=story_h,
                num_stories=num_stories,
                floor_mass=floor_mass,
                seismic_coeff=seismic_coeff,
                stiffness_reduction=stiffness_reduction,
                beam_stiffness_factor=beam_stiffness_factor,
            )
            drift_y, period_y = self._analyze_direction(
                ops=ops,
                wall_lengths=walls_x,
                beam_lengths=beams_y,
                wall_t=wall_t,
                beam_b=beam_b,
                beam_h=beam_h,
                e=e,
                story_h=story_h,
                num_stories=num_stories,
                floor_mass=floor_mass,
                seismic_coeff=seismic_coeff,
                stiffness_reduction=stiffness_reduction,
                beam_stiffness_factor=beam_stiffness_factor,
            )
            if drift_x is None and drift_y is None:
                return self._failed_result("analysis_not_converged_after_calibration")

            drift = max(v for v in [drift_x, drift_y] if v is not None)
            period = max(v for v in [period_x, period_y] if v is not None)

        shear_weight_ratio = seismic_coeff

        total_wall_area = wall_t * sum(walls_x + walls_y)
        avg_wall_area = total_wall_area / max(len(topology["shearwalls"]), 1)
        avg_wall_axial = (mass_total * 9.81) / max(len(topology["shearwalls"]), 1)
        axial_ratio = avg_wall_axial / max(fc * 1000.0 * avg_wall_area, 1.0e-6)

        drift_margin = self.constraints["drift_limit"] - drift
        axial_margin = self.constraints["axial_ratio_limit"] - axial_ratio
        shear_margin = shear_weight_ratio - self.constraints["shear_weight_ratio_min"]
        feasible = 1.0 if (drift_margin >= 0.0 and axial_margin >= 0.0 and shear_margin >= 0.0) else 0.0

        return {
            "drift": drift,
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
            "mass_dead_t": mass_dead,
            "mass_live_t": mass_live,
            "mass_total_t": mass_total,
            "calib_k_factor": calib_k,
            "calib_seismic_factor": calib_s,
            "calib_mass_dead_factor": mass_calib_dead,
            "calib_mass_live_factor": mass_calib_live,
        }

    def _analyze_direction(
        self,
        ops,
        wall_lengths: List[float],
        beam_lengths: List[float],
        wall_t: float,
        beam_b: float,
        beam_h: float,
        e: float,
        story_h: float,
        num_stories: int,
        floor_mass: float,
        seismic_coeff: float,
        stiffness_reduction: float,
        beam_stiffness_factor: float,
    ) -> Tuple[float, float]:
        if not wall_lengths:
            return None, None

        # Story lateral stiffness from wall + coupling beam contributions.
        i_wall = sum(wall_t * (lw**3) / 12.0 for lw in wall_lengths)
        i_beam = len(beam_lengths) * beam_b * (beam_h**3) / 12.0
        i_eff = max(stiffness_reduction * (i_wall + beam_stiffness_factor * i_beam), 1.0e-5)
        k_story = max(12.0 * e * i_eff / (story_h**3), 1.0)  # kN/m

        ops.wipe()
        ops.model("basic", "-ndm", 1, "-ndf", 1)

        for i in range(num_stories + 1):
            tag = i + 1
            ops.node(tag, i * story_h)
            if i == 0:
                ops.fix(tag, 1)
            else:
                ops.mass(tag, floor_mass)

        e_truss = 1.0e6
        mat_tag = 1
        ops.uniaxialMaterial("Elastic", mat_tag, e_truss)
        for i in range(1, num_stories + 1):
            area = max(k_story * story_h / e_truss, 1.0e-9)
            ops.element("truss", i, i, i + 1, area, mat_tag)

        lambdas = ops.eigen(1)
        if not lambdas:
            return None, None
        lam = float(lambdas[0])
        if lam <= 0.0:
            return None, None
        period = 2.0 * math.pi / math.sqrt(lam)

        total_weight = floor_mass * 9.81 * num_stories
        base_shear = seismic_coeff * total_weight
        idx_sum = num_stories * (num_stories + 1) / 2.0

        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for i in range(1, num_stories + 1):
            fi = base_shear * (i / idx_sum)
            ops.load(i + 1, fi)

        ops.system("BandGeneral")
        ops.numberer("RCM")
        ops.constraints("Plain")
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            return None, None

        drift_max = 0.0
        for i in range(1, num_stories + 1):
            u1 = float(ops.nodeDisp(i)[0])
            u2 = float(ops.nodeDisp(i + 1)[0])
            drift_i = abs((u2 - u1) / story_h)
            drift_max = max(drift_max, drift_i)

        return drift_max, period

    @staticmethod
    def _estimate_mass_ton(
        floor_area: float,
        total_wall_len: float,
        total_beam_len: float,
        wall_t: float,
        beam_b: float,
        beam_h: float,
        slab_t: float,
        density: float,
        dead: float,
        live: float,
        live_mass_factor: float,
        self_weight_scale: float,
        story_h: float,
        num_stories: int,
    ) -> Tuple[float, float, float]:
        # floor self-weight from wall, beam, slab volumes
        vol_wall = total_wall_len * wall_t * story_h
        vol_beam = total_beam_len * beam_b * beam_h
        vol_slab = floor_area * slab_t
        self_mass_floor = self_weight_scale * density * (vol_wall + vol_beam + vol_slab) / 1000.0

        dead_mass_floor = dead * floor_area / 9.81
        live_mass_floor = live_mass_factor * live * floor_area / 9.81

        dead_mass = (self_mass_floor + dead_mass_floor) * num_stories
        live_mass = live_mass_floor * num_stories
        return dead_mass, live_mass, dead_mass + live_mass

    @staticmethod
    def _split_members(members: List[Dict]) -> Tuple[List[float], List[float]]:
        x_list: List[float] = []
        y_list: List[float] = []
        tol = 1.0e-6
        for m in members:
            s = m.get("start", [0.0, 0.0])
            e = m.get("end", [0.0, 0.0])
            dx = float(e[0]) - float(s[0])
            dy = float(e[1]) - float(s[1])
            length = float(m.get("length", 0.0)) / 1000.0
            if length <= 0.0:
                continue
            if abs(dy) <= tol and abs(dx) > tol:
                x_list.append(length)
            elif abs(dx) <= tol and abs(dy) > tol:
                y_list.append(length)
        return x_list, y_list

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
            "cost_index": 1.0e9,
            "feasible": 0.0,
            "analysis_failed": 1.0,
            "failure_reason": reason,
            "drift_x": -1.0,
            "drift_y": -1.0,
            "period_x": -1.0,
            "period_y": -1.0,
            "mass_dead_t": -1.0,
            "mass_live_t": -1.0,
            "mass_total_t": -1.0,
            "calib_k_factor": -1.0,
            "calib_seismic_factor": -1.0,
            "calib_mass_dead_factor": -1.0,
            "calib_mass_live_factor": -1.0,
        }
