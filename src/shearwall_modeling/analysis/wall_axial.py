import openseespy.opensees as ops

from ..builders.base import AnalysisModelContext
from .member_forces import collect_wall_axial_metrics, extract_beam_force_tuple, extract_wall_force_tuple
from .results import GravityCaseResult


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
