import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.shearwall_modeling.geometry.scaling import load_and_scale_input
from src.shearwall_modeling.parametric import (
    DatasetGenerationConfig,
    ParametricModelParams,
    analyze_parametric_model,
)

from ..core.contracts import EvaluationResult, OptimizationProblem, VariableSpec


@dataclass(frozen=True)
class ShearWallObjectiveConfig:
    steel_weight: float = 1.0
    concrete_weight: float = 1.0
    infeasible_penalty: float = 1.0e12


@dataclass(frozen=True)
class ShearWallConstraintConfig:
    require_analysis_feasible: bool = True
    require_design_passed: bool = True


DEFAULT_DECISION_SPACE: dict[str, list[Any]] = {
    "tw_bot": [200, 250, 300, 350, 400],
    "tw_mid": [160, 180, 200, 250, 300],
    "tw_top": [160, 180, 200, 250],
    "hb_main": [400, 500, 550, 600, 650, 700],
    "bb_main": [200, 250, 300, 350],
    "hb_sec": [300, 400, 450, 500],
    "bb_sec": [200, 250, 300],
    "conc_bot": ["C30", "C35", "C40", "C45", "C50"],
}

DEFAULT_FIXED_PARAMS: dict[str, Any] = {
    "N": 28,
    "hs": 120,
    "h_story": 2.9,
    "intensity": 8.0,
    "site_class": "II",
    "seismic_group": 1,
}


class ShearWallOptimizationProblem(OptimizationProblem):
    def __init__(
        self,
        layout_path: str | Path,
        analysis_cfg: DatasetGenerationConfig,
        fixed_params: dict[str, Any] | None = None,
        decision_space: dict[str, list[Any]] | None = None,
        objective_cfg: ShearWallObjectiveConfig | None = None,
        constraint_cfg: ShearWallConstraintConfig | None = None,
        seed: int = 42,
    ):
        self.layout_path = Path(layout_path)
        self.analysis_cfg = analysis_cfg
        self.fixed_params = {**DEFAULT_FIXED_PARAMS, **(fixed_params or {})}
        self.decision_space = decision_space or DEFAULT_DECISION_SPACE
        self.objective_cfg = objective_cfg or ShearWallObjectiveConfig()
        self.constraint_cfg = constraint_cfg or ShearWallConstraintConfig()
        self.rng = random.Random(seed)
        self._cache: dict[tuple[Any, ...], EvaluationResult] = {}

        self._variables = [
            VariableSpec(name=name, domain=domain) for name, domain in self.decision_space.items()
        ]
        self.input_data, self.geom_scale = load_and_scale_input(
            json_path=self.layout_path,
            input_unit_scale_to_m=analysis_cfg.input_unit_scale_to_m,
            enable_auto_scale=analysis_cfg.enable_auto_scale,
            low=analysis_cfg.scale_low,
            high=analysis_cfg.scale_high,
            seed=analysis_cfg.scale_seed,
            manual_factor=analysis_cfg.manual_scale_factor,
        )

    @property
    def variables(self) -> list[VariableSpec]:
        return self._variables

    def sample(self) -> dict[str, Any]:
        return {spec.name: self.rng.choice(spec.domain) for spec in self._variables}

    def repair(self, x: dict[str, Any]) -> dict[str, Any]:
        fixed = dict(x)
        fixed["tw_mid"] = min(int(fixed["tw_mid"]), int(fixed["tw_bot"]))
        fixed["tw_top"] = min(int(fixed["tw_top"]), int(fixed["tw_mid"]))
        return fixed

    def evaluate(self, x: dict[str, Any]) -> EvaluationResult:
        repaired = self.repair(x)
        key = tuple((name, repaired[name]) for name in sorted(repaired.keys()))
        if key in self._cache:
            return self._cache[key]

        params = self._to_parametric_params(repaired)
        analysis_result = analyze_parametric_model(self.input_data, params, self.analysis_cfg)

        objective_raw = (
            self.objective_cfg.concrete_weight * analysis_result.material_concrete_kg
            + self.objective_cfg.steel_weight * analysis_result.material_steel_kg
        )

        constraints: dict[str, float] = {
            "not_converged": 0.0 if analysis_result.converged else 1.0,
            "analysis_unfeasible": (
                0.0
                if (analysis_result.feasible or not self.constraint_cfg.require_analysis_feasible)
                else 1.0
            ),
            "design_failed": (
                0.0
                if (analysis_result.design_passed or not self.constraint_cfg.require_design_passed)
                else 1.0
            ),
        }
        feasible = all(v <= 0.0 for v in constraints.values())

        objective = objective_raw
        if not feasible:
            objective += self.objective_cfg.infeasible_penalty * sum(constraints.values())

        metrics = {
            "geom_scale": self.geom_scale,
            **asdict(params),
            **asdict(analysis_result),
            "objective_raw": objective_raw,
            "objective": objective,
        }
        out = EvaluationResult(
            objective=objective,
            feasible=feasible,
            constraints=constraints,
            metrics=metrics,
        )
        self._cache[key] = out
        return out

    def encode(self, x: dict[str, Any]) -> list[float]:
        encoded: list[float] = []
        for spec in self._variables:
            idx = spec.domain.index(x[spec.name])
            denom = max(1, len(spec.domain) - 1)
            encoded.append(idx / denom)
        return encoded

    def decode(self, vec: list[float]) -> dict[str, Any]:
        if len(vec) != len(self._variables):
            raise ValueError("Vector length mismatch.")
        decoded: dict[str, Any] = {}
        for i, spec in enumerate(self._variables):
            v = min(1.0, max(0.0, float(vec[i])))
            idx = int(round(v * max(1, len(spec.domain) - 1)))
            decoded[spec.name] = spec.domain[idx]
        return self.repair(decoded)

    def _to_parametric_params(self, decision: dict[str, Any]) -> ParametricModelParams:
        merged = {**self.fixed_params, **decision}
        merged.setdefault("conc_mid", merged["conc_bot"])
        merged.setdefault("conc_top", merged["conc_bot"])
        return ParametricModelParams(
            N=int(merged["N"]),
            tw_bot=int(merged["tw_bot"]),
            tw_mid=int(merged["tw_mid"]),
            tw_top=int(merged["tw_top"]),
            hb_main=int(merged["hb_main"]),
            bb_main=int(merged["bb_main"]),
            hb_sec=int(merged["hb_sec"]),
            bb_sec=int(merged["bb_sec"]),
            hs=int(merged["hs"]),
            conc_bot=str(merged["conc_bot"]),
            conc_mid=str(merged["conc_mid"]),
            conc_top=str(merged["conc_top"]),
            intensity=float(merged["intensity"]),
            site_class=str(merged["site_class"]),
            seismic_group=int(merged["seismic_group"]),
            h_story=float(merged["h_story"]),
        )
