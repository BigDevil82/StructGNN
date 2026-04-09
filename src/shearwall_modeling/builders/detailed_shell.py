import math
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from logging import Logger

import openseespy.opensees as ops

from ..core.config import MaterialConfig, ModelConfig, StoryProfile
from ..core.domain import BeamRole, FEMInput
from ..geometry.mass_estimation import estimate_floor_area, estimate_structural_self_mass_per_floor
from .base import BeamElementUnit, ModelBuildResult, StructuralModelBuilder, WallBaseCheckUnit, WallStoryElementUnit


def _beam_section_props(width: float, depth: float) -> tuple[float, float, float, float]:
    area = width * depth
    # With geomTransf(..., 0,0,1) for floor beams, local z is vertical.
    # Therefore strong-axis inertia (b*h^3/12) should be Iy, not Iz.
    iy = width * (depth**3) / 12.0
    iz = depth * (width**3) / 12.0
    j = iy + iz
    return area, j, iy, iz


def _material_props(material: MaterialConfig) -> tuple[float, float, float]:
    e = float(material.E)
    g = float(material.G)
    nu = 0.2
    if abs(g) > 1.0e-12:
        nu_from_eg = e / (2.0 * g) - 1.0
        nu = min(0.45, max(0.05, nu_from_eg))
    return e, g, nu


class OpenSeesModelContext(AbstractContextManager["OpenSeesModelContext"]):
    """Manage OpenSees global model lifecycle and entity creation."""

    def __init__(self, coord_tol: float = 1.0e-3) -> None:
        self.coord_tol = coord_tol
        self.next_node_tag = 1
        self.next_element_tag = 1
        self.node_cache: dict[tuple[float, float, float], int] = {}
        self.node_coords: dict[int, tuple[float, float, float]] = {}

    def __enter__(self) -> "OpenSeesModelContext":
        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if exc_type is not None:
            ops.wipe()

    def _coord_key(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        q = self.coord_tol
        return (round(x / q) * q, round(y / q) * q, round(z / q) * q)

    def create_node(self, x: float, y: float, z: float) -> int:
        key = self._coord_key(x, y, z)
        if key in self.node_cache:
            return self.node_cache[key]

        tag = self.create_free_node(x, y, z)
        self.node_cache[key] = tag
        return tag

    def create_free_node(self, x: float, y: float, z: float) -> int:
        tag = self.next_node_tag
        self.next_node_tag += 1
        ops.node(tag, x, y, z)
        self.node_coords[tag] = (x, y, z)
        return tag

    def create_element(self, element_type: str, *args) -> int:
        tag = self.next_element_tag
        self.next_element_tag += 1
        ops.element(element_type, tag, *args)
        return tag

    def next_free_node_tag(self) -> int:
        return self.next_node_tag


@dataclass
class _ElementBuildResult:
    node_coords: dict[int, tuple[float, float, float]]
    floor_nodes: dict[int, set[int]]
    wall_base_nodes: list[list[int]]
    shell_count: int
    beam_count: int
    beam_count_by_role: dict[BeamRole, int]
    beam_element_units: list[BeamElementUnit]
    wall_story_element_units: list[WallStoryElementUnit]


class DetailedShellBuilder(StructuralModelBuilder):
    """Refined FE-like builder:

    - Shear walls: ShellMITC4 elements on extruded wall strips.
    - Beams: elasticBeamColumn elements at each floor.
    - Floors: rigid diaphragm constraints with one master node per story.
    """

    name = "detailed_shell"

    def __init__(
        self,
        logger: Logger,
        wall_mesh_size_m: float = 2.5,
        coord_tol: float = 1.0e-3,
    ) -> None:
        self.wall_mesh_size_m = wall_mesh_size_m
        self.coord_tol = coord_tol
        self.logger = logger
        self.model: OpenSeesModelContext | None = None

    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        self.logger.info("Building detailed shell model...")
        if not input_data.walls:
            raise ValueError("detailed_shell builder requires at least one shear wall.")

        with OpenSeesModelContext(coord_tol=self.coord_tol) as model:
            self.model = model
            profiles = config.resolve_story_profiles()
            z_levels = [0.0] + [p.z_top for p in profiles]

            floor_area = estimate_floor_area(input_data)
            floor_masses, total_load_mass, total_self_mass = self._calc_floor_masses(
                input_data=input_data,
                story_profiles=profiles,
                floor_area=floor_area,
            )
            total_structure_mass = sum(floor_masses)

            beam_transf_tag = 999
            ops.geomTransf("Linear", beam_transf_tag, 0.0, 0.0, 1.0)
            shell_section_by_story = self._create_shell_sections(profiles)
            element_result = self._build_shell_and_beam_elements(
                input_data=input_data,
                story_profiles=profiles,
                z_levels=z_levels,
                shell_section_by_story=shell_section_by_story,
                beam_transf_tag=beam_transf_tag,
                num_stories=config.num_stories,
            )
            master_nodes = self._create_story_masters(
                story_profiles=profiles,
                floor_plan_nodes=element_result.floor_nodes,
                node_coords=element_result.node_coords,
                floor_masses=floor_masses,
            )
            bottom_wall_thickness = profiles[0].section.wall_thickness
            wall_base_units = [
                WallBaseCheckUnit(
                    wall_id=i + 1,
                    base_nodes=base_nodes,
                    length=input_data.walls[i].length,
                    thickness=bottom_wall_thickness,
                )
                for i, base_nodes in enumerate(element_result.wall_base_nodes)
            ]
            floor_story_nodes = [
                sorted(element_result.floor_nodes[story]) for story in range(1, config.num_stories + 1)
            ]

            self.logger.info(
                f"Detailed model built: stories={config.num_stories}, walls={len(input_data.walls)}, "
                f"beams={len(input_data.beams)} (primary={len(input_data.beams_by_role(BeamRole.PRIMARY))}, "
                f"secondary={len(input_data.beams_by_role(BeamRole.SECONDARY))}), "
                f"shellElems={element_result.shell_count}, beamElems={element_result.beam_count} "
                f"(primary={element_result.beam_count_by_role[BeamRole.PRIMARY]}, "
                f"secondary={element_result.beam_count_by_role[BeamRole.SECONDARY]}), "
                f"floor_area~{floor_area:.2f} m^2, floor_mass_range=[{min(floor_masses)/1e3:.2f}, {max(floor_masses)/1e3:.2f}] t, "
                f"load_mass_total={total_load_mass/1e3:.2f} t, self_mass_total={total_self_mass/1e3:.2f} t, "
                f"total_mass={total_structure_mass/1e3:.2f} t ({total_structure_mass / 1e3:.2f} t)"
            )
            return ModelBuildResult(
                master_nodes=master_nodes,
                floor_area=floor_area,
                wall_base_units=wall_base_units,
                floor_story_nodes=floor_story_nodes,
                beam_element_units=element_result.beam_element_units,
                wall_story_element_units=element_result.wall_story_element_units,
            )

    def _calc_floor_masses(
        self, input_data: FEMInput, story_profiles: list[StoryProfile], floor_area: float
    ) -> tuple[list[float], float, float]:
        masses: list[float] = []
        total_load_mass = 0.0
        total_self_mass = 0.0

        for prf in story_profiles:
            load_mass = prf.mass_source.load_to_mass_per_area() * floor_area
            sec_w, sec_d = prf.section.get_beam_section(BeamRole.SECONDARY)
            self_mass_info = estimate_structural_self_mass_per_floor(
                input_data=input_data,
                story_height=prf.story_height,
                wall_thickness=prf.section.wall_thickness,
                primary_beam_width=prf.section.beam_width,
                primary_beam_depth=prf.section.beam_depth,
                secondary_beam_width=sec_w,
                secondary_beam_depth=sec_d,
                slab_thickness=prf.section.slab_thickness,
                density_kg_m3=prf.material.density_kg_m3,
                floor_area=floor_area,
            )
            self_mass = self_mass_info["total_self_mass"] if prf.mass_source.include_structural_self_weight else 0.0

            masses.append(load_mass + self_mass)
            total_load_mass += load_mass
            total_self_mass += self_mass

        return masses, total_load_mass, total_self_mass

    def _create_shell_sections(self, story_profiles: list[StoryProfile]) -> dict[int, int]:
        mat_tag = 1
        sec_tag = 1
        mat_cache: dict[tuple[float, float], int] = {}
        sections: dict[int, int] = {}

        for story, profile in enumerate(story_profiles, start=1):
            e, g, nu = _material_props(profile.material)
            mat_key = (e, g)
            if mat_key not in mat_cache:
                mat_cache[mat_key] = mat_tag
                ops.nDMaterial("ElasticIsotropic", mat_tag, e, nu)
                mat_tag += 1

            ops.section("PlateFiber", sec_tag, mat_cache[mat_key], profile.section.wall_thickness)
            sections[story] = sec_tag
            sec_tag += 1

        return sections

    def _build_shell_and_beam_elements(
        self,
        input_data: FEMInput,
        story_profiles: list[StoryProfile],
        z_levels: list[float],
        shell_section_by_story: dict[int, int],
        beam_transf_tag: int,
        num_stories: int,
    ) -> _ElementBuildResult:
        if self.model is None:
            raise RuntimeError("OpenSeesModelContext is not initialized.")
        fixed_base_nodes: set[int] = set()
        floor_nodes: dict[int, set[int]] = {story: set() for story in range(1, num_stories + 1)}
        wall_base_nodes: list[list[int]] = []
        wall_story_element_units: list[WallStoryElementUnit] = []

        shell_count = 0
        for wall in input_data.walls:
            if wall.length < 1.0e-6:
                continue
            div = max(1, int(math.ceil(wall.length / self.wall_mesh_size_m)))
            dx = (wall.end[0] - wall.start[0]) / div
            dy = (wall.end[1] - wall.start[1]) / div

            wall_grid: list[list[int]] = []
            for level in range(num_stories + 1):
                z = z_levels[level]
                row: list[int] = []
                for i in range(div + 1):
                    n = self.model.create_node(wall.start[0] + i * dx, wall.start[1] + i * dy, z)
                    row.append(n)
                    if level == 0:
                        if n not in fixed_base_nodes:
                            ops.fix(n, 1, 1, 1, 1, 1, 1)
                            fixed_base_nodes.add(n)
                    else:
                        floor_nodes[level].add(n)
                wall_grid.append(row)

            wall_base_nodes.append(sorted(set(wall_grid[0])))

            for level in range(num_stories):
                story_element_tags: list[int] = []
                for i in range(div):
                    n1 = wall_grid[level][i]
                    n2 = wall_grid[level][i + 1]
                    n3 = wall_grid[level + 1][i + 1]
                    n4 = wall_grid[level + 1][i]
                    tag = self.model.create_element("ShellMITC4", n1, n2, n3, n4, shell_section_by_story[level + 1])
                    story_element_tags.append(tag)
                    shell_count += 1
                wall_story_element_units.append(
                    WallStoryElementUnit(
                        wall_id=wall.m_id,
                        story=level + 1,
                        element_tags=story_element_tags,
                        bottom_nodes=sorted(set(wall_grid[level])),
                        top_nodes=sorted(set(wall_grid[level + 1])),
                        node_coords=dict(self.model.node_coords),
                        member=wall,
                        thickness=story_profiles[level].section.wall_thickness,
                        story_height=story_profiles[level].story_height,
                    )
                )

        beam_count = 0
        beam_count_by_role: dict[BeamRole, int] = {BeamRole.PRIMARY: 0, BeamRole.SECONDARY: 0}
        beam_element_units: list[BeamElementUnit] = []

        for story, profile in enumerate(story_profiles, start=1):
            z = profile.z_top
            e, g, _ = _material_props(profile.material)
            for beam in input_data.beams:
                beam_w, beam_d = profile.section.get_beam_section(beam.role)
                beam_area, beam_j, beam_iy, beam_iz = _beam_section_props(beam_w, beam_d)

                ni = self.model.create_node(beam.start[0], beam.start[1], z)
                nj = self.model.create_node(beam.end[0], beam.end[1], z)
                floor_nodes[story].add(ni)
                floor_nodes[story].add(nj)

                if ni == nj:
                    continue

                tag = self.model.create_element(
                    "elasticBeamColumn", ni, nj, beam_area, e, g, beam_j, beam_iy, beam_iz, beam_transf_tag
                )  # fmt: skip
                beam_count += 1
                beam_count_by_role[beam.role] += 1
                beam_element_units.append(
                    BeamElementUnit(
                        beam_id=beam.m_id,
                        story=story,
                        element_tag=tag,
                        member=beam,
                        role=beam.role,
                        width=beam_w,
                        depth=beam_d,
                        length=beam.length,
                    )
                )

        return _ElementBuildResult(
            node_coords=dict(self.model.node_coords),
            floor_nodes=floor_nodes,
            wall_base_nodes=wall_base_nodes,
            shell_count=shell_count,
            beam_count=beam_count,
            beam_count_by_role=beam_count_by_role,
            beam_element_units=beam_element_units,
            wall_story_element_units=wall_story_element_units,
        )

    def _create_story_masters(
        self,
        story_profiles: list[StoryProfile],
        floor_plan_nodes: dict[int, set[int]],
        node_coords: dict[int, tuple[float, float, float]],
        floor_masses: list[float],
    ) -> list[int]:
        if self.model is None:
            raise RuntimeError("OpenSeesModelContext is not initialized.")
        masters: list[int] = []

        for story, profile in enumerate(story_profiles, start=1):
            level_nodes = sorted(floor_plan_nodes[story])
            if not level_nodes:
                raise ValueError(f"No floor nodes were generated for story {story}.")

            z = profile.z_top
            com_x = sum(node_coords[n][0] for n in level_nodes) / len(level_nodes)
            com_y = sum(node_coords[n][1] for n in level_nodes) / len(level_nodes)

            master = self.model.create_free_node(com_x, com_y, z)

            floor_mass = floor_masses[story - 1]
            nodal_mass = floor_mass / len(level_nodes)
            rot_mass = sum(
                nodal_mass * ((node_coords[n][0] - com_x) ** 2 + (node_coords[n][1] - com_y) ** 2) for n in level_nodes
            )

            ops.mass(master, floor_mass, floor_mass, 0.0, 0.0, 0.0, rot_mass)
            ops.fix(master, 0, 0, 1, 1, 1, 0)
            if len(level_nodes) >= 2:
                ops.rigidDiaphragm(3, master, *level_nodes)
            masters.append(master)

        return masters
