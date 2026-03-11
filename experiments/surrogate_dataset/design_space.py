from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DesignSample:
    wall_thickness_mm: int
    beam_width_mm: int
    beam_depth_mm: int
    concrete_grade: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "wall_thickness_mm": float(self.wall_thickness_mm),
            "beam_width_mm": float(self.beam_width_mm),
            "beam_depth_mm": float(self.beam_depth_mm),
            "concrete_grade": float(self.concrete_grade),
        }


def _lhs_unit(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin Hypercube samples in [0, 1)."""
    result = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (perm + rng.random(n)) / n
    return result


def _pick_discrete(values: List[int], u: np.ndarray) -> np.ndarray:
    idx = np.minimum((u * len(values)).astype(int), len(values) - 1)
    return np.array([values[i] for i in idx], dtype=int)


def sample_designs(space_cfg: Dict, n_samples: int, seed: int) -> List[DesignSample]:
    rng = np.random.default_rng(seed)
    unit = _lhs_unit(n_samples, 4, rng)

    wall_t = _pick_discrete(space_cfg["wall_thickness_mm"], unit[:, 0])
    beam_b = _pick_discrete(space_cfg["beam_width_mm"], unit[:, 1])
    beam_h = _pick_discrete(space_cfg["beam_depth_mm"], unit[:, 2])
    fc = _pick_discrete(space_cfg["concrete_grade"], unit[:, 3])

    designs = []
    for i in range(n_samples):
        designs.append(
            DesignSample(
                wall_thickness_mm=int(wall_t[i]),
                beam_width_mm=int(beam_b[i]),
                beam_depth_mm=int(beam_h[i]),
                concrete_grade=int(fc[i]),
            )
        )
    return designs
