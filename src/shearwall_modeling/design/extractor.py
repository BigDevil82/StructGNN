import logging
import math

import openseespy.opensees as ops

from ..analysis.combinations import cqc, srss
from ..builders.base import BeamElementUnit, ModelBuildResult, WallStoryElementUnit
from ..core.config import ModelConfig
from .constants import CONCRETE_COMPRESSIVE_STRENGTH_MPA
from .results import BeamDesignDemand, WallDesignDemand


def _combine_modal_values(values: list[float], config: ModelConfig, eigen_values: list[float]) -> float:
    damping = [config.seismic.damping_ratio] * len(values)
    scale_factors = [1.0] * len(values)
    if config.seismic.combination_method.upper() == "CQC":
        return cqc(values, eigen_values, damping, scale_factors)
    return srss(values, scale_factors)


class DesignDemandExtractor:
    def __init__(self, build_result: ModelBuildResult, config: ModelConfig, logger: logging.Logger):
        self.build_result = build_result
        self.config = config
        self.logger = logger

    def extract(self) -> tuple[list[BeamDesignDemand], list[WallDesignDemand]]:
        gravity_beams, gravity_walls = self._run_gravity_case()
        seismic_beams, seismic_walls = self._run_response_spectrum_case()
        combined_beams = self._combine_beam_demands(gravity_beams, seismic_beams)
        combined_walls = self._combine_wall_demands(gravity_walls, seismic_walls)
        return combined_beams, combined_walls

    def _run_gravity_case(
        self,
    ) -> tuple[
        dict[tuple[int, int], tuple[float, float, float]], dict[tuple[int, int], tuple[float, float, float]]
    ]:
        ts_tag = 81001
        pat_tag = 81001
        floor_area = self.build_result.floor_area
        story_profiles = self.config.resolve_story_profiles()

        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)
        for story_index, floor_nodes in enumerate(self.build_result.floor_story_nodes):
            floor_force_n = (
                (
                    story_profiles[story_index].mass_source.dead_kpa
                    + 0.5 * story_profiles[story_index].mass_source.live_kpa
                )
                * 1000.0
                * floor_area
            )
            nodal_force = floor_force_n / len(floor_nodes)
            for node in floor_nodes:
                ops.load(node, 0.0, 0.0, -nodal_force, 0.0, 0.0, 0.0)

        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Newton")
        ops.integrator("LoadControl", 1.0)
        ops.analysis("Static")
        if ops.analyze(1) != 0:
            raise RuntimeError("Gravity analysis failed while extracting reinforcement design demands.")

        beam_demands = {
            (unit.beam_id, unit.story): self._extract_beam_force_tuple(unit)
            for unit in self.build_result.beam_element_units
        }
        wall_demands = {
            (unit.wall_id, unit.story): self._extract_wall_force_tuple(unit, direction=None)
            for unit in self.build_result.wall_story_element_units
        }
        return beam_demands, wall_demands

    def _run_response_spectrum_case(
        self,
    ) -> tuple[
        dict[tuple[int, int], tuple[float, float, float]],
        dict[tuple[int, int], tuple[float, float, float, float]],
    ]:
        ops.wipeAnalysis()
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.system("UmfPack")
        ops.test("NormDispIncr", 1.0e-6, 20)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.0)
        ops.analysis("Static")

        eigen_values = ops.eigen(
            "-genBandArpack", min(self.config.num_modes, len(self.build_result.master_nodes) * 2)
        )
        if isinstance(eigen_values, (int, float)):
            eigen_values = [float(eigen_values)]
        else:
            eigen_values = [float(value) for value in eigen_values if float(value) > 1.0e-12]

        ops.modalProperties("-return")

        beam_modal: dict[tuple[int, int], dict[str, dict[str, list[float]]]] = {}
        wall_modal: dict[tuple[int, int], dict[str, dict[str, list[float]]]] = {}
        for dir_idx, dir_name in ((1, "X"), (2, "Y")):
            for mode_index in range(1, len(eigen_values) + 1):
                ops.responseSpectrumAnalysis(
                    dir_idx,
                    "-Tn",
                    *self.config.seismic.periods,
                    "-Sa",
                    *self.config.seismic.spectral_accel,
                    "-mode",
                    mode_index,
                )
                for unit in self.build_result.beam_element_units:
                    key = (unit.beam_id, unit.story)
                    pos_m, neg_m, shear_v = self._extract_beam_force_tuple(unit)
                    store = beam_modal.setdefault(
                        key,
                        {
                            "X": {"pos": [], "neg": [], "shear": []},
                            "Y": {"pos": [], "neg": [], "shear": []},
                        },
                    )[dir_name]
                    store["pos"].append(pos_m)
                    store["neg"].append(neg_m)
                    store["shear"].append(shear_v)
                for unit in self.build_result.wall_story_element_units:
                    key = (unit.wall_id, unit.story)
                    axial_n, moment_m, shear_v = self._extract_wall_force_tuple(unit, direction=dir_name)
                    store = wall_modal.setdefault(
                        key,
                        {
                            "X": {"axial": [], "moment": [], "shear": []},
                            "Y": {"axial": [], "moment": [], "shear": []},
                        },
                    )[dir_name]
                    store["axial"].append(axial_n)
                    store["moment"].append(moment_m)
                    store["shear"].append(shear_v)

        beam_demands = {}
        for key, directional_store in beam_modal.items():
            positive_moment = max(
                _combine_modal_values(direction_store["pos"], self.config, eigen_values)
                for direction_store in directional_store.values()
            )
            negative_moment = max(
                _combine_modal_values(direction_store["neg"], self.config, eigen_values)
                for direction_store in directional_store.values()
            )
            shear_force = max(
                _combine_modal_values(direction_store["shear"], self.config, eigen_values)
                for direction_store in directional_store.values()
            )
            beam_demands[key] = (positive_moment, negative_moment, shear_force)

        wall_demands = {}
        for unit in self.build_result.wall_story_element_units:
            key = (unit.wall_id, unit.story)
            store = wall_modal.get(
                key,
                {
                    "X": {"axial": [], "moment": [], "shear": []},
                    "Y": {"axial": [], "moment": [], "shear": []},
                },
            )
            wall_demands[key] = (
                max(
                    _combine_modal_values(direction_store["axial"], self.config, eigen_values)
                    for direction_store in store.values()
                ),
                max(
                    _combine_modal_values(direction_store["moment"], self.config, eigen_values)
                    for direction_store in store.values()
                ),
                max(
                    _combine_modal_values(direction_store["shear"], self.config, eigen_values)
                    for direction_store in store.values()
                ),
                0.0,
            )
        return beam_demands, wall_demands

    def _extract_beam_force_tuple(self, unit: BeamElementUnit) -> tuple[float, float, float]:
        force = [float(value) for value in ops.eleForce(unit.element_tag)]
        if len(force) < 12:
            return 0.0, 0.0, 0.0
        end_moments = [force[4], force[10]]
        positive_moment = max([value for value in end_moments if value > 0.0] + [0.0])
        negative_moment = max([abs(value) for value in end_moments if value < 0.0] + [0.0])
        shear = max(abs(force[2]), abs(force[8]))
        return positive_moment, negative_moment, shear

    def _extract_wall_force_tuple(
        self, unit: WallStoryElementUnit, direction: str | None
    ) -> tuple[float, float, float]:
        node_force_sum: dict[int, list[float]] = {}
        node_force_count: dict[int, int] = {}
        for tag in unit.element_tags:
            raw = ops.eleResponse(tag, "forces")
            values = [float(value) for value in raw] if raw else [float(value) for value in ops.eleForce(tag)]
            if len(values) < 24:
                continue
            for local_index, node in enumerate((unit.bottom_nodes + unit.top_nodes)[:4]):
                start = local_index * 6
                force_slice = values[start : start + 6]
                node_force_sum.setdefault(node, [0.0] * 6)
                node_force_count[node] = node_force_count.get(node, 0) + 1
                for idx, value in enumerate(force_slice):
                    node_force_sum[node][idx] += value

        if not node_force_sum:
            return 0.0, 0.0, 0.0

        centroid_x = 0.5 * (unit.member.start.x + unit.member.end.x)
        centroid_y = 0.5 * (unit.member.start.y + unit.member.end.y)
        axis_x = unit.member.end.x - unit.member.start.x
        axis_y = unit.member.end.y - unit.member.start.y
        axis_len = max(math.hypot(axis_x, axis_y), 1.0e-9)
        axis_x /= axis_len
        axis_y /= axis_len

        axial_force = 0.0
        bending_moment = 0.0
        shear_force = 0.0
        for node in unit.bottom_nodes:
            values = [
                item / max(node_force_count.get(node, 1), 1) for item in node_force_sum.get(node, [0.0] * 6)
            ]
            x, y, _ = unit.node_coords[node]
            axial_force += abs(values[2])
            arm_m = (x - centroid_x) * axis_x + (y - centroid_y) * axis_y
            bending_moment += abs(values[2] * arm_m)
            if direction == "X":
                shear_force += abs(values[0])
            elif direction == "Y":
                shear_force += abs(values[1])
            else:
                shear_force += max(abs(values[0]), abs(values[1]))
        return axial_force, bending_moment, shear_force

    def _combine_beam_demands(
        self,
        gravity: dict[tuple[int, int], tuple[float, float, float]],
        seismic: dict[tuple[int, int], tuple[float, float, float]],
    ) -> list[BeamDesignDemand]:
        demand_by_key: dict[tuple[int, int], BeamDesignDemand] = {}
        for unit in self.build_result.beam_element_units:
            key = (unit.beam_id, unit.story)
            gravity_pos, gravity_neg, gravity_shear = gravity.get(key, (0.0, 0.0, 0.0))
            seismic_pos, seismic_neg, seismic_shear = seismic.get(key, (0.0, 0.0, 0.0))
            demand_by_key[key] = BeamDesignDemand(
                beam_id=unit.beam_id,
                story=unit.story,
                width_m=unit.width,
                depth_m=unit.depth,
                length_m=unit.length,
                positive_moment_n_m=1.2 * abs(gravity_pos) + 1.3 * abs(seismic_pos),
                negative_moment_n_m=1.2 * abs(gravity_neg) + 1.3 * abs(seismic_neg),
                shear_n=1.2 * abs(gravity_shear) + 1.3 * abs(seismic_shear),
            )
        return list(demand_by_key.values())

    def _combine_wall_demands(
        self,
        gravity: dict[tuple[int, int], tuple[float, float, float]],
        seismic: dict[tuple[int, int], tuple[float, float, float, float]],
    ) -> list[WallDesignDemand]:
        demands: list[WallDesignDemand] = []
        story_profiles = {profile.story: profile for profile in self.config.resolve_story_profiles()}
        for unit in self.build_result.wall_story_element_units:
            key = (unit.wall_id, unit.story)
            gravity_axial, gravity_moment, gravity_shear = gravity.get(key, (0.0, 0.0, 0.0))
            seismic_axial, seismic_moment, seismic_shear, _ = seismic.get(key, (0.0, 0.0, 0.0, 0.0))
            area_m2 = unit.member.length * unit.thickness
            combined_axial = 1.2 * abs(gravity_axial) + 1.3 * abs(seismic_axial)
            concrete_grade = story_profiles[unit.story].material.concrete_grade.strip().upper()
            fc_pa = CONCRETE_COMPRESSIVE_STRENGTH_MPA[concrete_grade] * 1.0e6
            axial_ratio = combined_axial / max(area_m2 * fc_pa, 1.0e-9)
            demands.append(
                WallDesignDemand(
                    wall_id=unit.wall_id,
                    story=unit.story,
                    thickness_m=unit.thickness,
                    length_m=unit.member.length,
                    story_height_m=unit.story_height,
                    axial_force_n=combined_axial,
                    moment_n_m=1.2 * abs(gravity_moment) + 1.3 * abs(seismic_moment),
                    shear_n=1.2 * abs(gravity_shear) + 1.3 * abs(seismic_shear),
                    axial_ratio=axial_ratio,
                    is_bottom_reinforced_zone=unit.story <= 2,
                )
            )
        return demands
