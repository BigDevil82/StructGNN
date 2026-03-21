import math

import openseespy.opensees as ops

from ..config import ModelConfig
from ..domain import FEMInput, PlanMember
from ..geometry import estimate_floor_area
from .base import ModelBuildResult, StructuralModelBuilder


def _safe_vecxz_from_wall(member: PlanMember) -> tuple[float, float, float]:
    dx = member.end[0] - member.start[0]
    dy = member.end[1] - member.start[1]
    l = math.hypot(dx, dy)
    if l <= 1e-9:
        return (1.0, 0.0, 0.0)
    return (-dy / l, dx / l, 0.0)


def _wall_section_props(length: float, thickness: float) -> tuple[float, float, float, float]:
    area = length * thickness
    iz = thickness * (length**3) / 12.0
    iy = length * (thickness**3) / 12.0
    j = iy + iz
    return area, j, iy, iz


def _beam_section_props(width: float, depth: float) -> tuple[float, float, float, float]:
    area = width * depth
    iz = width * (depth**3) / 12.0
    iy = depth * (width**3) / 12.0
    j = iy + iz
    return area, j, iy, iz


class EquivalentFrameBuilder(StructuralModelBuilder):
    name = "equivalent_frame"

    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        e = config.material.E
        g = config.material.G
        floor_area = estimate_floor_area(input_data)
        floor_mass = config.mass_per_area * floor_area

        beam_area, beam_j, beam_iy, beam_iz = _beam_section_props(
            config.section.beam_width, config.section.beam_depth
        )

        node_tag = 1
        elem_tag = 1
        transf_tag = 1000
        beam_transf = 999
        ops.geomTransf("Linear", beam_transf, 0.0, 0.0, 1.0)

        plan_node_to_base_node: dict[tuple[float, float], int] = {}
        master_nodes: list[int] = []

        for story in range(1, config.num_stories + 1):
            z_base = (story - 1) * config.story_height
            z_top = story * config.story_height
            story_nodes: dict[tuple[float, float], int] = {}
            slave_nodes: list[int] = []
            linked_slaves: set[int] = set()

            for wall in input_data.walls:
                wall_len = max(wall.length, 1e-6)
                area, wall_j, wall_iy, wall_iz = _wall_section_props(wall_len, config.section.wall_thickness)
                cx = 0.5 * (wall.start[0] + wall.end[0])
                cy = 0.5 * (wall.start[1] + wall.end[1])

                if story == 1:
                    base_tag = node_tag
                    ops.node(base_tag, cx, cy, z_base)
                    ops.fix(base_tag, 1, 1, 1, 1, 1, 1)
                    node_tag += 1
                    plan_node_to_base_node[(cx, cy)] = base_tag
                else:
                    base_tag = plan_node_to_base_node[(cx, cy)]

                top_tag = node_tag
                ops.node(top_tag, cx, cy, z_top)
                node_tag += 1
                plan_node_to_base_node[(cx, cy)] = top_tag

                vecxz = _safe_vecxz_from_wall(wall)
                ops.geomTransf("Linear", transf_tag, *vecxz)

                ops.element(
                    "elasticBeamColumn",
                    elem_tag,
                    base_tag,
                    top_tag,
                    area,
                    e,
                    g,
                    wall_j,
                    wall_iy,
                    wall_iz,
                    transf_tag,
                )
                elem_tag += 1
                transf_tag += 1
                slave_nodes.append(top_tag)

                for endpoint in (wall.start, wall.end):
                    if endpoint not in story_nodes:
                        n = node_tag
                        ops.node(n, endpoint[0], endpoint[1], z_top)
                        node_tag += 1
                        story_nodes[endpoint] = n
                        slave_nodes.append(n)

                    s = story_nodes[endpoint]
                    if s not in linked_slaves:
                        ops.rigidLink("beam", top_tag, s)
                        linked_slaves.add(s)

            for beam in input_data.beams:
                if beam.start not in story_nodes:
                    n = node_tag
                    ops.node(n, beam.start[0], beam.start[1], z_top)
                    node_tag += 1
                    story_nodes[beam.start] = n
                    slave_nodes.append(n)
                if beam.end not in story_nodes:
                    n = node_tag
                    ops.node(n, beam.end[0], beam.end[1], z_top)
                    node_tag += 1
                    story_nodes[beam.end] = n
                    slave_nodes.append(n)

                ni = story_nodes[beam.start]
                nj = story_nodes[beam.end]
                if ni == nj:
                    continue

                ops.element(
                    "elasticBeamColumn",
                    elem_tag,
                    ni,
                    nj,
                    beam_area,
                    e,
                    g,
                    beam_j,
                    beam_iy,
                    beam_iz,
                    beam_transf,
                )
                elem_tag += 1

            if not story_nodes:
                raise ValueError("No floor nodes generated from walls/beams. Check input JSON contents.")

            com_x = sum(p[0] for p in story_nodes.keys()) / len(story_nodes)
            com_y = sum(p[1] for p in story_nodes.keys()) / len(story_nodes)

            master = node_tag
            ops.node(master, com_x, com_y, z_top)
            node_tag += 1

            rot_mass = floor_mass * (max(1.0, floor_area)) / 12.0
            ops.mass(master, floor_mass, floor_mass, 0.0, 0.0, 0.0, rot_mass)
            ops.fix(master, 0, 0, 1, 1, 1, 0)

            unique_slaves = sorted(set(slave_nodes))
            if len(unique_slaves) >= 2:
                ops.rigidDiaphragm(3, master, *unique_slaves)

            master_nodes.append(master)

        print(
            f"Model built: {config.num_stories} stories, walls={len(input_data.walls)}, beams={len(input_data.beams)}, "
            f"floor_area~{floor_area:.2f} m^2"
        )
        return ModelBuildResult(master_nodes=master_nodes, floor_area=floor_area)
