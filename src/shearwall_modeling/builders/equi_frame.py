import math
from logging import Logger

import openseespy.opensees as ops

from ..core.config import ModelConfig
from ..core.domain import BeamRole, FEMInput, PlanMember
from ..geometry.mass_estimation import estimate_floor_area, estimate_structural_self_mass_per_floor
from .base import (
    BeamElementUnit,
    ModelBuildResult,
    StructuralModelBuilder,
    WallBaseCheckUnit,
    WallStoryElementUnit,
)


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
    j = (length * thickness**3) / 3.0  # 矩形截面抗扭常数
    return area, j, iy, iz


def _beam_section_props(width: float, depth: float) -> tuple[float, float, float, float]:
    area = width * depth
    iz = width * (depth**3) / 12.0
    iy = depth * (width**3) / 12.0
    j = (width * depth**3) / 3.0  # 矩形截面抗扭常数
    return area, j, iy, iz


class EquivalentFrameBuilder(StructuralModelBuilder):
    name = "equivalent_frame"

    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger

    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        if self.logger is not None:
            self.logger.info("Building equivalent frame model...")
        else:
            print("Building equivalent frame model...")

        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        profiles = config.resolve_story_profiles()
        if not profiles:
            raise ValueError("No story profiles resolved from model config.")

        floor_area = estimate_floor_area(input_data)
        floor_masses: list[float] = []
        floor_load_masses: list[float] = []
        floor_self_masses: list[float] = []
        self_mass_info_last: dict[str, float] = {"wall_mass": 0.0, "beam_mass": 0.0, "slab_mass": 0.0}

        for profile in profiles:
            load_mass = profile.mass_source.load_to_mass_per_area() * floor_area
            sec_w, sec_d = profile.section.get_beam_section(BeamRole.SECONDARY)
            self_mass_info = estimate_structural_self_mass_per_floor(
                input_data=input_data,
                story_height=profile.story_height,
                wall_thickness=profile.section.wall_thickness,
                primary_beam_width=profile.section.beam_width,
                primary_beam_depth=profile.section.beam_depth,
                secondary_beam_width=sec_w,
                secondary_beam_depth=sec_d,
                slab_thickness=profile.section.slab_thickness,
                density_kg_m3=profile.material.density_kg_m3,
                floor_area=floor_area,
            )
            self_mass = (
                self_mass_info["total_self_mass"]
                if profile.mass_source.include_structural_self_weight
                else 0.0
            )
            floor_masses.append(load_mass + self_mass)
            floor_load_masses.append(load_mass)
            floor_self_masses.append(self_mass)
            self_mass_info_last = self_mass_info

        total_structure_mass = sum(floor_masses)

        node_tag = 1
        elem_tag = 1
        transf_tag = 1000
        beam_transf = 999
        ops.geomTransf("Linear", beam_transf, 0.0, 0.0, 1.0)

        plan_node_to_base_node: dict[tuple[float, float], int] = {}
        master_nodes: list[int] = []
        wall_base_units: list[WallBaseCheckUnit] = []
        floor_story_nodes: list[list[int]] = []
        beam_element_units: list[BeamElementUnit] = []
        wall_story_element_units: list[WallStoryElementUnit] = []

        for story in range(1, config.num_stories + 1):
            profile = profiles[story - 1]
            e = float(profile.material.E)
            g = float(profile.material.G)
            z_base = profile.z_bottom
            z_top = profile.z_top
            story_nodes: dict[tuple[float, float], int] = {}
            slave_nodes: list[int] = []
            linked_slaves: set[int] = set()
            story_floor_nodes: set[int] = set()

            for wall in input_data.walls:
                wall_len = max(wall.length, 1e-6)
                area, wall_j, wall_iy, wall_iz = _wall_section_props(wall_len, profile.section.wall_thickness)
                cx = 0.5 * (wall.start[0] + wall.end[0])
                cy = 0.5 * (wall.start[1] + wall.end[1])

                if story == 1:
                    base_tag = node_tag
                    ops.node(base_tag, cx, cy, z_base)
                    ops.fix(base_tag, 1, 1, 1, 1, 1, 1)
                    node_tag += 1
                    plan_node_to_base_node[(cx, cy)] = base_tag
                    wall_base_units.append(
                        WallBaseCheckUnit(
                            wall_id=wall.m_id,
                            base_nodes=[base_tag],
                            length=wall.length,
                            thickness=profile.section.wall_thickness,
                        )
                    )
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
                story_floor_nodes.add(top_tag)

                wall_story_element_units.append(
                    WallStoryElementUnit(
                        wall_id=wall.m_id,
                        story=story,
                        element_tags=[elem_tag - 1],
                        bottom_nodes=[base_tag],
                        top_nodes=[top_tag],
                        node_coords={
                            base_tag: (
                                ops.nodeCoord(base_tag, 1),
                                ops.nodeCoord(base_tag, 2),
                                ops.nodeCoord(base_tag, 3),
                            ),
                            top_tag: (
                                ops.nodeCoord(top_tag, 1),
                                ops.nodeCoord(top_tag, 2),
                                ops.nodeCoord(top_tag, 3),
                            ),
                        },
                        member=wall,
                        thickness=profile.section.wall_thickness,
                        story_height=profile.story_height,
                    )
                )

                for endpoint in (wall.start, wall.end):
                    if endpoint not in story_nodes:
                        n = node_tag
                        ops.node(n, endpoint[0], endpoint[1], z_top)
                        node_tag += 1
                        story_nodes[endpoint] = n
                        slave_nodes.append(n)
                        story_floor_nodes.add(n)

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
                    story_floor_nodes.add(n)
                if beam.end not in story_nodes:
                    n = node_tag
                    ops.node(n, beam.end[0], beam.end[1], z_top)
                    node_tag += 1
                    story_nodes[beam.end] = n
                    slave_nodes.append(n)
                    story_floor_nodes.add(n)

                ni = story_nodes[beam.start]
                nj = story_nodes[beam.end]
                if ni == nj:
                    continue

                beam_w, beam_d = profile.section.get_beam_section(beam.role)
                beam_area, beam_j, beam_iy, beam_iz = _beam_section_props(beam_w, beam_d)

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
                beam_element_units.append(
                    BeamElementUnit(
                        beam_id=beam.m_id,
                        story=story,
                        element_tag=elem_tag,
                        member=beam,
                        role=beam.role,
                        width=beam_w,
                        depth=beam_d,
                        length=beam.length,
                    )
                )
                elem_tag += 1

            if not story_nodes:
                raise ValueError("No floor nodes generated from walls/beams. Check input JSON contents.")

            com_x = sum(p[0] for p in story_nodes.keys()) / len(story_nodes)
            com_y = sum(p[1] for p in story_nodes.keys()) / len(story_nodes)

            master = node_tag
            ops.node(master, com_x, com_y, z_top)
            node_tag += 1

            unique_slaves = sorted(set(slave_nodes) - linked_slaves)
            story_mass = floor_masses[story - 1]
            nodal_mass = story_mass / max(len(unique_slaves), 1)
            rot_mass = sum(
                nodal_mass * ((ops.nodeCoord(n, 1) - com_x) ** 2 + (ops.nodeCoord(n, 2) - com_y) ** 2)
                for n in unique_slaves
            )
            ops.mass(master, story_mass, story_mass, 0.0, 0.0, 0.0, rot_mass)
            ops.fix(master, 0, 0, 1, 1, 1, 0)

            if len(unique_slaves) >= 1:
                ops.rigidDiaphragm(3, master, *unique_slaves)

            master_nodes.append(master)
            floor_story_nodes.append(sorted(story_floor_nodes))

        msg = (
            f"Model built: {config.num_stories} stories, walls={len(input_data.walls)}, beams={len(input_data.beams)}, "
            f"floor_area~{floor_area:.2f} m^2, floor_mass_range=[{min(floor_masses):.2f}, {max(floor_masses):.2f}] kg, "
            f"self_mass_breakdown_last(kg): wall={self_mass_info_last.get('wall_mass', 0.0):.2f}, "
            f"beam={self_mass_info_last.get('beam_mass', 0.0):.2f}, slab={self_mass_info_last.get('slab_mass', 0.0):.2f}, "
            f"total_mass={total_structure_mass:.2f} kg ({total_structure_mass / 1000.0:.3f} t)"
        )
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

        return ModelBuildResult(
            master_nodes=master_nodes,
            floor_area=floor_area,
            floor_load_masses=floor_load_masses,
            floor_self_masses=floor_self_masses,
            wall_base_units=wall_base_units,
            floor_story_nodes=floor_story_nodes,
            beam_element_units=beam_element_units,
            wall_story_element_units=wall_story_element_units,
        )
