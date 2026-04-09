import math

import openseespy.opensees as ops

from ..builders.base import AnalysisModelContext, WallStoryElementUnit
from ..core.constants import MIN_WALL_LEN_CHECK_THRESHOLD_M, concrete_fc_pa
from .results import WallAxialMetric


def extract_beam_force_tuple(element_tag: int) -> tuple[float, float, float]:
    force = [float(value) for value in ops.eleForce(element_tag)]
    if len(force) < 12:
        return 0.0, 0.0, 0.0
    end_moments = [force[4], force[10]]
    positive_moment = max([value for value in end_moments if value > 0.0] + [0.0])
    negative_moment = max([abs(value) for value in end_moments if value < 0.0] + [0.0])
    shear = max(abs(force[2]), abs(force[8]))
    return positive_moment, negative_moment, shear


def extract_wall_force_tuple(unit: WallStoryElementUnit, dir_name: str | None) -> tuple[float, float, float]:
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
        values = [item / max(node_force_count.get(node, 1), 1) for item in node_force_sum.get(node, [0.0] * 6)]
        x, y, _ = unit.node_coords[node]
        axial_force += abs(values[2])
        arm_m = (x - centroid_x) * axis_x + (y - centroid_y) * axis_y
        bending_moment += abs(values[2] * arm_m)
        if dir_name == "X":
            shear_force += abs(values[0])
        elif dir_name == "Y":
            shear_force += abs(values[1])
        else:
            shear_force += max(abs(values[0]), abs(values[1]))
    return axial_force, bending_moment, shear_force


def collect_wall_axial_metrics(context: AnalysisModelContext) -> list[WallAxialMetric]:
    base_nodes = sorted({node for unit in context.wall_base_units for node in unit.base_nodes})
    node_reactions = {node: ops.nodeReaction(node, 3) for node in base_nodes}
    node_share = {node: 0 for node in base_nodes}
    for unit in context.wall_base_units:
        for node in unit.base_nodes:
            node_share[node] += 1

    fc_pa = concrete_fc_pa(context.story_profiles[0].material.concrete_grade)
    ratio_limit = context.config.seismic.axial_compression_ratio_limit
    metrics: list[WallAxialMetric] = []
    for unit in context.wall_base_units:
        if unit.length < MIN_WALL_LEN_CHECK_THRESHOLD_M:
            continue
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
