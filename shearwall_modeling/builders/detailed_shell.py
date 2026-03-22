import math

import openseespy.opensees as ops

from ..config import ModelConfig
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


class DetailedShellBuilder(StructuralModelBuilder):
    """Refined FE-like builder:

    - Shear walls: ShellMITC4 elements on extruded wall strips.
    - Beams: elasticBeamColumn elements at each floor.
    - Floors: rigid diaphragm constraints with one master node per story.
    """

    name = "detailed_shell"

    def __init__(self, wall_mesh_size_m: float = 2.5, coord_tol: float = 1.0e-6) -> None:
        self.wall_mesh_size_m = wall_mesh_size_m
        self.coord_tol = coord_tol

    def build(self, input_data: FEMInput, config: ModelConfig) -> ModelBuildResult:
        print("Building detailed shell model...")
        if not input_data.walls:
            raise ValueError("detailed_shell builder requires at least one shear wall.")

        ops.wipe()
        ops.model("basic", "-ndm", 3, "-ndf", 6)

        e = config.material.E
        g = config.material.G
        nu = 0.2
        if abs(g) > 1.0e-12:
            nu_from_eg = e / (2.0 * g) - 1.0
            nu = min(0.45, max(0.05, nu_from_eg))

        wall_thickness = config.section.wall_thickness
        beam_area, beam_j, beam_iy, beam_iz = _beam_section_props(
            config.section.beam_width, config.section.beam_depth
        )

        floor_area = estimate_floor_area(input_data)
        load_mass_per_area = config.mass_source.load_to_mass_per_area()
        load_mass_per_floor = load_mass_per_area * floor_area
        self_mass_info = estimate_structural_self_mass_per_floor(
            input_data=input_data,
            story_height=config.story_height,
            wall_thickness=config.section.wall_thickness,
            beam_width=config.section.beam_width,
            beam_depth=config.section.beam_depth,
            slab_thickness=config.section.slab_thickness,
            density_kg_m3=config.material.density_kg_m3,
            floor_area=floor_area,
        )
        self_mass_per_floor = (
            self_mass_info["total_mass"] if config.mass_source.include_structural_self_weight else 0.0
        )
        floor_mass = load_mass_per_floor + self_mass_per_floor
        total_structure_mass = floor_mass * config.num_stories

        mat_tag = 1
        shell_sec_tag = 1
        beam_transf_tag = 999
        ops.nDMaterial("ElasticIsotropic", mat_tag, e, nu)
        ops.section("PlateFiber", shell_sec_tag, mat_tag, wall_thickness)
        ops.geomTransf("Linear", beam_transf_tag, 0.0, 0.0, 1.0)

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
                z = level * config.story_height
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
                    ops.element("ShellMITC4", elem_tag, n1, n2, n3, n4, shell_sec_tag)
                    elem_tag += 1
                    shell_count += 1

        beam_edges: set[tuple[int, int]] = set()
        beam_count = 0
        for story in range(1, config.num_stories + 1):
            z = story * config.story_height
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
        for story in range(1, config.num_stories + 1):
            level_nodes = sorted(floor_plan_nodes[story])
            if not level_nodes:
                raise ValueError(f"No floor nodes were generated for story {story}.")

            z = story * config.story_height
            com_x = sum(node_coords[n][0] for n in level_nodes) / len(level_nodes)
            com_y = sum(node_coords[n][1] for n in level_nodes) / len(level_nodes)

            master = node_tag
            ops.node(master, com_x, com_y, z)
            node_tag += 1

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

        print(
            f"Detailed model built: stories={config.num_stories}, walls={len(input_data.walls)}, "
            f"beams={len(input_data.beams)}, shellElems={shell_count}, beamElems={beam_count}, "
            f"floor_area~{floor_area:.2f} m^2, load_mass_per_area={load_mass_per_area:.2f} kg/m^2, "
            f"load_mass_per_floor={load_mass_per_floor:.2f} kg, self_mass_per_floor={self_mass_per_floor:.2f} kg, "
            f"self_mass_breakdown(kg): wall={self_mass_info['wall_mass']:.2f}, beam={self_mass_info['beam_mass']:.2f}, slab={self_mass_info['slab_mass']:.2f}, "
            f"floor_mass={floor_mass:.2f} kg, "
            f"total_mass={total_structure_mass:.2f} kg ({total_structure_mass / 1000.0:.3f} t)"
        )
        return ModelBuildResult(master_nodes=master_nodes, floor_area=floor_area)
