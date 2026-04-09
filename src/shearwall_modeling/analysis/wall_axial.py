import openseespy.opensees as ops

from ..core.constants import concrete_fc_pa
from .response_spectrum import AnalysisModelContext
from .results import WallAxialMetric


class WallAxialCompressionChecker:
    def __init__(self, context: AnalysisModelContext):
        self.context = context

    def _story_gravity_force_n(self, story_index: int) -> float:
        profile = self.context.story_profiles[story_index]
        return self.context.floor_masses[story_index] * profile.mass_source.gravity

    def check(self) -> list[WallAxialMetric]:
        snapshot = self.context.build_result.analysis_snapshot
        if snapshot is not None:
            return snapshot.wall_axial_metrics

        ts_tag = 70001
        pat_tag = 70001
        floor_gravity_forces = [self._story_gravity_force_n(story_index) for story_index in range(self.context.num_stories)]

        ops.timeSeries("Linear", ts_tag)
        ops.pattern("Plain", pat_tag, ts_tag)
        for story_index, floor_nodes in enumerate(self.context.floor_story_nodes):
            nodal_force = floor_gravity_forces[story_index] / len(floor_nodes)
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
            raise RuntimeError("Static gravity case failed during wall axial check.")

        ops.reactions()
        base_nodes = sorted({node for unit in self.context.wall_base_units for node in unit.base_nodes})
        node_reactions = {node: ops.nodeReaction(node, 3) for node in base_nodes}
        node_share = {node: 0 for node in base_nodes}
        for unit in self.context.wall_base_units:
            for node in unit.base_nodes:
                node_share[node] += 1

        fc_pa = concrete_fc_pa(self.context.story_profiles[0].material.concrete_grade)
        ratio_limit = self.context.config.seismic.axial_compression_ratio_limit
        metrics: list[WallAxialMetric] = []
        for unit in self.context.wall_base_units:
            axial_force_n = sum(node_reactions[node] / node_share[node] for node in unit.base_nodes)
            area_m2 = unit.length * unit.thickness
            stress_pa = axial_force_n / area_m2
            axial_ratio = stress_pa / fc_pa
            metrics.append(
                WallAxialMetric(
                    wall_id=unit.wall_id,
                    axial_force_n=axial_force_n,
                    area_m2=area_m2,
                    axial_stress_mpa=stress_pa / 1.0e6,
                    axial_ratio=axial_ratio,
                    ratio_limit=ratio_limit,
                    is_passed=axial_ratio <= ratio_limit,
                )
            )
        return metrics
