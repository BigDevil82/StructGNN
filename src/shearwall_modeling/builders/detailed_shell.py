import math
from logging import Logger

import openseespy.opensees as ops

from ..config import MaterialConfig, ModelConfig
from ..domain import FEMInput, PlanMember
from ..geometry import estimate_floor_area, estimate_structural_self_mass_per_floor
from .base import ModelBuildResult, StructuralModelBuilder


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
        coord_tol: float = 1.0e-6,
    ) -> None:
        self.wall_mesh_size_m = wall_mesh_size_m
        self.coord_tol = coord_tol
        self.logger = logger

    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        self.logger.info("Building detailed shell model...")
        if not input_data.walls:
            raise ValueError("detailed_shell builder requires at least one shear wall.")

        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        story_profiles = config.resolve_story_profiles()
        z_levels = [0.0] + [profile.z_top for profile in story_profiles]

        floor_area = estimate_floor_area(input_data)
        floor_masses: list[float] = []
        total_load_mass = 0.0
        total_self_mass = 0.0
        for profile in story_profiles:
            load_mass_per_area = profile.mass_source.load_to_mass_per_area()
            load_mass_per_floor = load_mass_per_area * floor_area
            self_mass_info = estimate_structural_self_mass_per_floor(
                input_data=input_data,
                story_height=profile.story_height,
                wall_thickness=profile.section.wall_thickness,
                beam_width=profile.section.beam_width,
                beam_depth=profile.section.beam_depth,
                slab_thickness=profile.section.slab_thickness,
                density_kg_m3=profile.material.density_kg_m3,
                floor_area=floor_area,
            )
            self_mass = (
                self_mass_info["total_mass"] if profile.mass_source.include_structural_self_weight else 0.0
            )
            floor_masses.append(load_mass_per_floor + self_mass)
            total_load_mass += load_mass_per_floor
            total_self_mass += self_mass

        total_structure_mass = sum(floor_masses)

        mat_tag = 1
        shell_sec_tag = 1
        beam_transf_tag = 999
        ops.geomTransf("Linear", beam_transf_tag, 0.0, 0.0, 1.0)

        material_tag_cache: dict[tuple[float, float], int] = {}

        shell_section_by_story: dict[int, int] = {}
        section_tag = shell_sec_tag
        for story, profile in enumerate(story_profiles, start=1):
            e, g, nu = _material_props(profile.material)
            mat_key = (e, g)
            story_mat_tag = material_tag_cache.get(mat_key)
            if story_mat_tag is None:
                story_mat_tag = mat_tag
                mat_tag += 1
                ops.nDMaterial("ElasticIsotropic", story_mat_tag, e, nu)
                material_tag_cache[mat_key] = story_mat_tag

            ops.section("PlateFiber", section_tag, story_mat_tag, profile.section.wall_thickness)
            shell_section_by_story[story] = section_tag
            section_tag += 1

        node_tag = 1
        elem_tag = 1

        node_cache: dict[tuple[float, float, float], int] = {}
        node_coords: dict[int, tuple[float, float, float]] = {}
        fixed_base_nodes: set[int] = set()

        def _coord_key(x: float, y: float, z: float) -> tuple[float, float, float]:
            q = self.coord_tol
            return (round(x / q) * q, round(y / q) * q, round(z / q) * q)

        def get_or_create_node(x: float, y: float, z: float) -> int:
            nonlocal node_tag
            key = _coord_key(x, y, z)
            if key in node_cache:
                return node_cache[key]
            tag = node_tag
            ops.node(tag, x, y, z)
            node_cache[key] = tag
            node_coords[tag] = (x, y, z)
            node_tag += 1
            return tag

        floor_plan_nodes: dict[int, set[int]] = {story: set() for story in range(1, config.num_stories + 1)}

        shell_count = 0
        for wall in input_data.walls:
            wall_len = max(wall.length, 1.0e-9)
            if wall_len < 1.0e-6:
                continue

            div = max(1, int(math.ceil(wall_len / self.wall_mesh_size_m)))
            dx = (wall.end[0] - wall.start[0]) / div
            dy = (wall.end[1] - wall.start[1]) / div

            wall_grid: list[list[int]] = []
            for level in range(0, config.num_stories + 1):
                z = z_levels[level]
                row: list[int] = []
                for i in range(div + 1):
                    x = wall.start[0] + i * dx
                    y = wall.start[1] + i * dy
                    n = get_or_create_node(x, y, z)
                    row.append(n)
                    if level == 0:
                        if n not in fixed_base_nodes:
                            ops.fix(n, 1, 1, 1, 1, 1, 1)
                            fixed_base_nodes.add(n)
                    else:
                        floor_plan_nodes[level].add(n)
                wall_grid.append(row)

            for level in range(config.num_stories):
                for i in range(div):
                    n1 = wall_grid[level][i]
                    n2 = wall_grid[level][i + 1]
                    n3 = wall_grid[level + 1][i + 1]
                    n4 = wall_grid[level + 1][i]
                    story = level + 1
                    ops.element("ShellMITC4", elem_tag, n1, n2, n3, n4, shell_section_by_story[story])
                    elem_tag += 1
                    shell_count += 1

        beam_edges: set[tuple[int, int]] = set()
        beam_count = 0
        for story, profile in enumerate(story_profiles, start=1):
            z = profile.z_top
            e, g, _ = _material_props(profile.material)
            beam_area, beam_j, beam_iy, beam_iz = _beam_section_props(
                profile.section.beam_width,
                profile.section.beam_depth,
            )
            for beam in input_data.beams:
                ni = get_or_create_node(beam.start[0], beam.start[1], z)
                nj = get_or_create_node(beam.end[0], beam.end[1], z)
                floor_plan_nodes[story].add(ni)
                floor_plan_nodes[story].add(nj)
                if ni == nj:
                    continue

                edge = (min(ni, nj), max(ni, nj))
                if edge in beam_edges:
                    continue
                beam_edges.add(edge)

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
                    beam_transf_tag,
                )
                elem_tag += 1
                beam_count += 1

        master_nodes: list[int] = []
        for story, profile in enumerate(story_profiles, start=1):
            level_nodes = sorted(floor_plan_nodes[story])
            if not level_nodes:
                raise ValueError(f"No floor nodes were generated for story {story}.")

            z = profile.z_top
            com_x = sum(node_coords[n][0] for n in level_nodes) / len(level_nodes)
            com_y = sum(node_coords[n][1] for n in level_nodes) / len(level_nodes)

            master = node_tag
            ops.node(master, com_x, com_y, z)
            node_tag += 1

            floor_mass = floor_masses[story - 1]
            nodal_mass = floor_mass / len(level_nodes)
            rot_mass = sum(
                nodal_mass * ((node_coords[n][0] - com_x) ** 2 + (node_coords[n][1] - com_y) ** 2)
                for n in level_nodes
            )
            ops.mass(master, floor_mass, floor_mass, 0.0, 0.0, 0.0, rot_mass)
            ops.fix(master, 0, 0, 1, 1, 1, 0)

            if len(level_nodes) >= 2:
                ops.rigidDiaphragm(3, master, *level_nodes)

            master_nodes.append(master)

        self.logger.info(
            f"Detailed model built: stories={config.num_stories}, walls={len(input_data.walls)}, "
            f"beams={len(input_data.beams)}, shellElems={shell_count}, beamElems={beam_count}, "
            f"floor_area~{floor_area:.2f} m^2, floor_mass_range=[{min(floor_masses)/1e3:.2f}, {max(floor_masses)/1e3:.2f}] t, "
            f"load_mass_total={total_load_mass/1e3:.2f} t, self_mass_total={total_self_mass/1e3:.2f} t, "
            f"total_mass={total_structure_mass/1e3:.2f} t ({total_structure_mass / 1e3:.2f} t)"
        )
        return ModelBuildResult(master_nodes=master_nodes, floor_area=floor_area)
