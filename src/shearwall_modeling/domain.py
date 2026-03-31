import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PlanMember:
    start: tuple[float, float]
    end: tuple[float, float]

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    def scaled(self, factor: float) -> "PlanMember":
        return PlanMember(
            start=(self.start[0] * factor, self.start[1] * factor),
            end=(self.end[0] * factor, self.end[1] * factor),
        )


@dataclass
class FEMInput:
    walls: list[PlanMember]
    beams: list[PlanMember]
    slabs: list[list[tuple[float, float]]]

    @classmethod
    def from_json(cls, json_path: Path, xy_scale_to_m: float = 0.001) -> "FEMInput":
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        walls = [
            PlanMember(
                start=(float(w["start"][0]) * xy_scale_to_m, float(w["start"][1]) * xy_scale_to_m),
                end=(float(w["end"][0]) * xy_scale_to_m, float(w["end"][1]) * xy_scale_to_m),
            )
            for w in data.get("shearwalls", [])
        ]
        beams = [
            PlanMember(
                start=(float(b["start"][0]) * xy_scale_to_m, float(b["start"][1]) * xy_scale_to_m),
                end=(float(b["end"][0]) * xy_scale_to_m, float(b["end"][1]) * xy_scale_to_m),
            )
            for b in data.get("beams", [])
        ]
        slabs = [
            [
                (float(p[0]) * xy_scale_to_m, float(p[1]) * xy_scale_to_m)
                for p in slab
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            for slab in data.get("slabs", [])
            if isinstance(slab, list)
        ]
        return cls(walls=walls, beams=beams, slabs=slabs)

    def all_members(self) -> list[PlanMember]:
        return self.walls + self.beams

    def scaled(self, factor: float) -> "FEMInput":
        return FEMInput(
            walls=[w.scaled(factor) for w in self.walls],
            beams=[b.scaled(factor) for b in self.beams],
            slabs=[[(x * factor, y * factor) for x, y in slab] for slab in self.slabs],
        )
