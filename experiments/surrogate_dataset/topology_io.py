import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class TopologySummary:
    topology_id: str
    source_path: str
    num_nodes: int
    num_walls: int
    num_beams: int
    total_wall_length_mm: float
    total_beam_length_mm: float
    plan_span_x_mm: float
    plan_span_y_mm: float
    wall_density: float

    def to_feature_dict(self) -> Dict[str, float]:
        return {
            "num_nodes": float(self.num_nodes),
            "num_walls": float(self.num_walls),
            "num_beams": float(self.num_beams),
            "total_wall_length_mm": self.total_wall_length_mm,
            "total_beam_length_mm": self.total_beam_length_mm,
            "plan_span_x_mm": self.plan_span_x_mm,
            "plan_span_y_mm": self.plan_span_y_mm,
            "wall_density": self.wall_density,
        }


def _span(coords: List[List[float]]) -> (float, float):
    if not coords:
        return 0.0, 0.0
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return float(max(xs) - min(xs)), float(max(ys) - min(ys))


def load_topology_summary(path: Path) -> TopologySummary:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    walls = data.get("shearwalls", [])
    beams = data.get("beams", [])

    coords = [[n["x"], n["y"]] for n in nodes if "x" in n and "y" in n]
    span_x, span_y = _span(coords)
    area = max(span_x * span_y, 1.0)

    total_wall = float(sum(w.get("length", 0.0) for w in walls))
    total_beam = float(sum(b.get("length", 0.0) for b in beams))

    topology_id = path.stem
    wall_density = total_wall / area

    return TopologySummary(
        topology_id=topology_id,
        source_path=str(path),
        num_nodes=len(nodes),
        num_walls=len(walls),
        num_beams=len(beams),
        total_wall_length_mm=total_wall,
        total_beam_length_mm=total_beam,
        plan_span_x_mm=span_x,
        plan_span_y_mm=span_y,
        wall_density=wall_density,
    )

