import logging

from ..analysis.response_spectrum import AnalysisModelContext, ResponseSpectrumAnalyzer
from ..builders.base import ModelBuildResult
from ..core.config import ModelConfig
from .constants import CONCRETE_COMPRESSIVE_STRENGTH_MPA
from .results import BeamDesignDemand, WallDesignDemand


class DesignDemandExtractor:
    def __init__(self, build_result: ModelBuildResult, config: ModelConfig, logger: logging.Logger):
        self.build_result = build_result
        self.config = config
        self.logger = logger
        context = AnalysisModelContext(build_result=build_result, config=config, logger=logger)
        self.analyzer = ResponseSpectrumAnalyzer(context)

    def extract(self) -> tuple[list[BeamDesignDemand], list[WallDesignDemand]]:
        snapshot = self._get_snapshot()
        combined_beams = self._combine_beam_demands(
            snapshot.gravity_beam_forces,
            snapshot.seismic_beam_forces,
        )
        combined_walls = self._combine_wall_demands(
            snapshot.gravity_wall_forces,
            snapshot.seismic_wall_forces,
        )
        return combined_beams, combined_walls

    def _get_snapshot(self):
        snapshot = self.build_result.analysis_snapshot
        if snapshot is None:
            snapshot = self.analyzer.build_snapshot()
            self.build_result.analysis_snapshot = snapshot
        return snapshot

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
