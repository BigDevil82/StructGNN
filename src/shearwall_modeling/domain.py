import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

Point2D = tuple[float, float]


class BeamRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"

    @classmethod
    def from_raw(cls, raw: Any) -> "BeamRole":
        if isinstance(raw, BeamRole):
            return raw
        if raw is None:
            return cls.PRIMARY
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized == cls.PRIMARY.value:
                return cls.PRIMARY
            if normalized == cls.SECONDARY.value:
                return cls.SECONDARY
        raise ValueError(f"Unsupported beam_role={raw}. Expected 'primary' or 'secondary'.")


def _parse_point(raw: Any, xy_scale_to_m: float) -> Point2D:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValueError("Point must be a list/tuple with at least 2 values.")
    return float(raw[0]) * xy_scale_to_m, float(raw[1]) * xy_scale_to_m


@dataclass
class PlanMember:
    start: Point2D
    end: Point2D

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
    beams: list["BeamMember"]
    slabs: list[list[Point2D]]

    @classmethod
    def from_json(cls, json_path: Path, xy_scale_to_m: float = 0.001) -> "FEMInput":
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        walls = [
            PlanMember(
                start=_parse_point(w["start"], xy_scale_to_m),
                end=_parse_point(w["end"], xy_scale_to_m),
            )
            for w in data.get("shearwalls", [])
        ]

        beams = [
            BeamMember(
                start=_parse_point(b["start"], xy_scale_to_m),
                end=_parse_point(b["end"], xy_scale_to_m),
                role=BeamRole.from_raw(b.get("beam_role")),
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

    def beams_by_role(self, role: BeamRole) -> list["BeamMember"]:
        return [beam for beam in self.beams if beam.role == role]

    def beam_length_by_role(self, role: BeamRole) -> float:
        return sum(beam.length for beam in self.beams if beam.role == role and beam.length > 1.0e-9)

    def scaled(self, factor: float) -> "FEMInput":
        return FEMInput(
            walls=[w.scaled(factor) for w in self.walls],
            beams=[b.scaled(factor) for b in self.beams],
            slabs=[[(x * factor, y * factor) for x, y in slab] for slab in self.slabs],
        )


@dataclass
class BeamMember(PlanMember):
    role: BeamRole = BeamRole.PRIMARY

    def scaled(self, factor: float) -> "BeamMember":
        return BeamMember(
            start=(self.start[0] * factor, self.start[1] * factor),
            end=(self.end[0] * factor, self.end[1] * factor),
            role=self.role,
        )
