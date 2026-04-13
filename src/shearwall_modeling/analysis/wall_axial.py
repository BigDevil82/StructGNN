import openseespy.opensees as ops

from ..builders.base import AnalysisModelContext
from .member_forces import collect_wall_axial_metrics, extract_beam_force_tuple, extract_wall_force_tuple
from .results import GravityCaseResult, ResponseSpectrumCaseResult


class GravityCaseAnalyzer:
    def __init__(self, context: AnalysisModelContext):
        self.context = context

    def _story_gravity_force_n(self, story_index: int) -> float:
        profile = self.context.story_profiles[story_index]
        return self.context.floor_masses[story_index] * profile.mass_source.gravity

    def _configure_gravity_analysis(self) -> None:
        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")

    def run(self) -> GravityCaseResult:
        ts_tag = 70001
        pat_tag = 70001
        floor_gravity_forces = [
            self._story_gravity_force_n(story_index) for story_index in range(self.context.num_stories)
        ]

        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)
        for story_index, floor_nodes in enumerate(self.context.floor_story_nodes):
            nodal_force = floor_gravity_forces[story_index] / len(floor_nodes)
            for node in floor_nodes:
                ops.load(node, 0.0, 0.0, -nodal_force, 0.0, 0.0, 0.0)

        self._configure_gravity_analysis()
        if ops.analyze(1) != 0:
            raise RuntimeError("Static gravity case failed during wall axial check.")

        gravity_beam_forces = {
            (unit.beam_id, unit.story): extract_beam_force_tuple(unit.element_tag)
            for unit in self.context.beam_element_units
        }
        gravity_wall_forces = {
            (unit.wall_id, unit.story): extract_wall_force_tuple(unit, dir_name=None)
            for unit in self.context.wall_story_element_units
        }
        ops.reactions()
        return GravityCaseResult(
            gravity_beam_forces=gravity_beam_forces,
            gravity_wall_forces=gravity_wall_forces,
            wall_axial_metrics=collect_wall_axial_metrics(self.context),
        )


class LinearSuperpositionAnalyzer:
    def __init__(
        self,
        context: AnalysisModelContext,
        wind_base_shear_ratio_x: float = 0.0,
        wind_base_shear_ratio_y: float = 0.0,
        wind_story_forces_x: list[float] | None = None,
        wind_story_forces_y: list[float] | None = None,
    ):
        self.context = context
        self.basic_cases = ["Dead", "Live", "WindX", "WindY"]
        self.wind_base_shear_ratio_x = max(float(wind_base_shear_ratio_x), 0.0)
        self.wind_base_shear_ratio_y = max(float(wind_base_shear_ratio_y), 0.0)
        self.wind_story_forces_x = wind_story_forces_x
        self.wind_story_forces_y = wind_story_forces_y

    def _configure_static_analysis(self) -> None:
        """配置线性静力分析（通用配置）"""
        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        # 因为是线弹性分析，算法用 Linear 即可，Newton 也没错但 Linear 更快
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")

    def _apply_story_vertical_loads(self, story_forces_n: list[float]) -> None:
        for story_index, floor_nodes in enumerate(self.context.floor_story_nodes):
            if not floor_nodes:
                continue
            nodal_force = story_forces_n[story_index] / len(floor_nodes)
            for node in floor_nodes:
                ops.load(node, 0.0, 0.0, -nodal_force, 0.0, 0.0, 0.0)

    def _story_dead_force_n(self, story_index: int) -> float:
        profile = self.context.story_profiles[story_index]
        return self.context.floor_area * profile.mass_source.dead_kpa * 1000.0

    def _story_live_force_n(self, story_index: int) -> float:
        profile = self.context.story_profiles[story_index]
        return self.context.floor_area * profile.mass_source.live_kpa * 1000.0

    def _gravity_story_weights_n(self) -> list[float]:
        return [
            self.context.floor_masses[story_index]
            * self.context.story_profiles[story_index].mass_source.gravity
            for story_index in range(self.context.num_stories)
        ]

    def _lateral_story_forces(
        self, base_shear_ratio: float, explicit_story_forces: list[float] | None
    ) -> list[float]:
        if explicit_story_forces is not None:
            if len(explicit_story_forces) != self.context.num_stories:
                raise ValueError("Length of explicit wind story forces must equal num_stories.")
            return [max(float(value), 0.0) for value in explicit_story_forces]

        if base_shear_ratio <= 0.0:
            return [0.0] * self.context.num_stories

        story_weights = self._gravity_story_weights_n()
        heights = [profile.z_top for profile in self.context.story_profiles]
        weighted = [max(story_weights[idx] * heights[idx], 0.0) for idx in range(self.context.num_stories)]
        total_weighted = sum(weighted)
        if total_weighted <= 1.0e-9:
            return [0.0] * self.context.num_stories

        base_shear_n = base_shear_ratio * sum(story_weights)
        return [base_shear_n * value / total_weighted for value in weighted]

    def _apply_lateral_load(self, story_forces_n: list[float], direction: str) -> None:
        for story_index, node in enumerate(self.context.master_nodes):
            force = story_forces_n[story_index]
            if direction == "X":
                ops.load(node, force, 0.0, 0.0, 0.0, 0.0, 0.0)
            elif direction == "Y":
                ops.load(node, 0.0, force, 0.0, 0.0, 0.0, 0.0)
            else:
                raise ValueError(f"Unsupported direction {direction}.")

    def _apply_dead_load(self) -> None:
        story_forces_n = [self._story_dead_force_n(idx) for idx in range(self.context.num_stories)]
        self._apply_story_vertical_loads(story_forces_n)

    def _apply_live_load(self) -> None:
        story_forces_n = [self._story_live_force_n(idx) for idx in range(self.context.num_stories)]
        self._apply_story_vertical_loads(story_forces_n)

    def _apply_wind_x_load(self) -> None:
        story_forces_n = self._lateral_story_forces(self.wind_base_shear_ratio_x, self.wind_story_forces_x)
        self._apply_lateral_load(story_forces_n, direction="X")

    def _apply_wind_y_load(self) -> None:
        story_forces_n = self._lateral_story_forces(self.wind_base_shear_ratio_y, self.wind_story_forces_y)
        self._apply_lateral_load(story_forces_n, direction="Y")

    def _extract_case_forces(self, wall_dir: str | None) -> tuple[
        dict[tuple[int, int], tuple[float, float, float]],
        dict[tuple[int, int], tuple[float, float, float]],
    ]:
        beam_forces = {
            (unit.beam_id, unit.story): extract_beam_force_tuple(unit.element_tag)
            for unit in self.context.beam_element_units
        }
        wall_forces = {
            (unit.wall_id, unit.story): extract_wall_force_tuple(unit, dir_name=wall_dir)
            for unit in self.context.wall_story_element_units
        }
        return beam_forces, wall_forces

    def _run_static_basic_case(self, case_name: str) -> tuple[
        dict[tuple[int, int], tuple[float, float, float]],
        dict[tuple[int, int], tuple[float, float, float]],
        list,
    ]:
        ts_tag = 71000 + self.basic_cases.index(case_name) + 1
        pat_tag = 71000 + self.basic_cases.index(case_name) + 1
        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)

        if case_name == "Dead":
            self._apply_dead_load()
            wall_dir = None
        elif case_name == "Live":
            self._apply_live_load()
            wall_dir = None
        elif case_name == "WindX":
            self._apply_wind_x_load()
            wall_dir = "X"
        elif case_name == "WindY":
            self._apply_wind_y_load()
            wall_dir = "Y"
        else:
            raise ValueError(f"Unsupported basic case {case_name}.")

        self._configure_static_analysis()
        if ops.analyze(1) != 0:
            raise RuntimeError(f"Static analysis failed for basic case: {case_name}")

        ops.reactions()
        beam_forces, wall_forces = self._extract_case_forces(wall_dir=wall_dir)
        wall_metrics = collect_wall_axial_metrics(self.context)
        ops.remove("loadPattern", pat_tag)
        ops.reset()
        return beam_forces, wall_forces, wall_metrics

    def run_basic_cases(self, rsa_result: ResponseSpectrumCaseResult | None = None) -> dict:
        """Run basic static cases and optionally append EqX/EqY from RSA directional results."""
        basic_results: dict[str, dict] = {}
        for case in self.basic_cases:
            beam_forces, wall_forces, wall_metrics = self._run_static_basic_case(case)
            basic_results[case] = {
                "beam_forces": beam_forces,
                "wall_forces": wall_forces,
                "wall_axial_metrics": wall_metrics,
            }

        if rsa_result is not None:
            beam_by_dir = rsa_result.seismic_beam_forces_by_dir or {}
            wall_by_dir = rsa_result.seismic_wall_forces_by_dir or {}
            has_directional = (
                "X" in beam_by_dir and "Y" in beam_by_dir and "X" in wall_by_dir and "Y" in wall_by_dir
            )
            if has_directional:
                basic_results["EqX"] = {
                    "beam_forces": beam_by_dir["X"],
                    "wall_forces": wall_by_dir["X"],
                    "wall_axial_metrics": [],
                }
                basic_results["EqY"] = {
                    "beam_forces": beam_by_dir["Y"],
                    "wall_forces": wall_by_dir["Y"],
                    "wall_axial_metrics": [],
                }
            else:
                self.context.logger.warning(
                    "RSA directional forces are unavailable; fallback to enveloped seismic forces for EqX/EqY."
                )
                basic_results["EqX"] = {
                    "beam_forces": rsa_result.seismic_beam_forces,
                    "wall_forces": rsa_result.seismic_wall_forces,
                    "wall_axial_metrics": [],
                }
                basic_results["EqY"] = {
                    "beam_forces": rsa_result.seismic_beam_forces,
                    "wall_forces": rsa_result.seismic_wall_forces,
                    "wall_axial_metrics": [],
                }

        return basic_results

    # ==========================================
    # 3. 荷载效应组合模块
    # ==========================================
    def combine_effects(self, basic_results: dict, combinations: list[dict]) -> dict:
        """
        在外部对基本工况效应进行矩阵组合
        combinations 示例:
        [
            {'name': '1.3D+1.5L', 'factors': {'Dead': 1.3, 'Live': 1.5}},
            {'name': '1.2D+1.4L+1.2WX', 'factors': {'Dead': 1.2, 'Live': 1.4, 'WindX': 1.2}}
        ]
        """
        combined_results = {}

        for combo in combinations:
            combo_name = combo["name"]
            factors = combo["factors"]

            combo_beam_forces: dict[tuple[int, int], tuple[float, float, float]] = {}
            combo_wall_forces: dict[tuple[int, int], tuple[float, float, float]] = {}

            beam_keys = {
                key for value in basic_results.values() for key in value.get("beam_forces", {}).keys()
            }
            wall_keys = {
                key for value in basic_results.values() for key in value.get("wall_forces", {}).keys()
            }

            for key in beam_keys:
                v0, v1, v2 = 0.0, 0.0, 0.0
                for case, factor in factors.items():
                    f0, f1, f2 = basic_results.get(case, {}).get("beam_forces", {}).get(key, (0.0, 0.0, 0.0))
                    v0 += abs(float(factor)) * abs(float(f0))
                    v1 += abs(float(factor)) * abs(float(f1))
                    v2 += abs(float(factor)) * abs(float(f2))
                combo_beam_forces[key] = (v0, v1, v2)

            for key in wall_keys:
                v0, v1, v2 = 0.0, 0.0, 0.0
                for case, factor in factors.items():
                    f0, f1, f2 = basic_results.get(case, {}).get("wall_forces", {}).get(key, (0.0, 0.0, 0.0))
                    v0 += abs(float(factor)) * abs(float(f0))
                    v1 += abs(float(factor)) * abs(float(f1))
                    v2 += abs(float(factor)) * abs(float(f2))
                combo_wall_forces[key] = (v0, v1, v2)

            combined_results[combo_name] = {
                "beam_forces": combo_beam_forces,
                "wall_forces": combo_wall_forces,
                # 如果还需要组合 wall_axial_metrics，可以用同样的 NumPy 累加逻辑
            }

        return combined_results
