import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.misc.parallel import run_batch
from src.shearwall_modeling.geometry.scaling import load_and_scale_input
from src.shearwall_modeling.parametric import (
    DatasetGenerationConfig,
    ParametricModelParams,
    analyze_parametric_model,
)

from ..core.contracts import EvaluationResult, OptimizationProblem, VariableSpec


@dataclass(frozen=True)
class ShearWallObjectiveConfig:
    concrete_price_per_kg: float = 0.0005
    steel_price_per_kg: float = 0.005
    infeasible_penalty: float = 1e6


@dataclass(frozen=True)
class ShearWallConstraintConfig:
    require_analysis_feasible: bool = True
    require_design_passed: bool = True


@dataclass(frozen=True)
class ShearWallLimitConfig:
    max_torsion_ratio: float = 1.5
    max_drift_ratio: float = 1.0 / 1000.0
    min_shear_weight_ratio: float = 0.016
    min_stiffness_ratio: float = 0.7
    max_period_ratio: float = 0.9


@dataclass(frozen=True)
class BatchEvaluateTask:
    request_index: int
    layout_path: str
    analysis_cfg: dict[str, Any]
    fixed_params: dict[str, Any]
    objective_cfg: dict[str, Any]
    constraint_cfg: dict[str, Any]
    limit_cfg: dict[str, Any]
    decision: dict[str, Any]


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
        limit_cfg: ShearWallLimitConfig | None = None,
        seed: int = 42,
    ):
        self.layout_path = Path(layout_path)
        self.analysis_cfg = analysis_cfg
        self.fixed_params = {**DEFAULT_FIXED_PARAMS, **(fixed_params or {})}
        self.decision_space = decision_space or DEFAULT_DECISION_SPACE
        self.objective_cfg = objective_cfg or ShearWallObjectiveConfig()
        self.constraint_cfg = constraint_cfg or ShearWallConstraintConfig()
        self.limit_cfg = limit_cfg or ShearWallLimitConfig()
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

        out = _evaluate_decision(
            input_data=self.input_data,
            geom_scale=self.geom_scale,
            decision=repaired,
            fixed_params=self.fixed_params,
            analysis_cfg=self.analysis_cfg,
            objective_cfg=self.objective_cfg,
            constraint_cfg=self.constraint_cfg,
            limit_cfg=self.limit_cfg,
        )
        self._cache[key] = out
        return out

    def evaluate_many(
        self, xs: list[dict[str, Any]], max_workers: int | None = None
    ) -> list[EvaluationResult]:
        if not xs:
            return []
        if not max_workers or max_workers <= 1 or len(xs) < 2:
            return [self.evaluate(x) for x in xs]

        results: list[EvaluationResult | None] = [None] * len(xs)
        uncached_items: list[tuple[int, dict[str, Any], tuple[Any, ...]]] = []

        for i, x in enumerate(xs):
            repaired = self.repair(x)
            key = tuple((name, repaired[name]) for name in sorted(repaired.keys()))
            cached = self._cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_items.append((i, repaired, key))

        if uncached_items:
            tasks = [
                BatchEvaluateTask(
                    request_index=item[0],
                    layout_path=str(self.layout_path),
                    analysis_cfg=asdict(self.analysis_cfg),
                    fixed_params=dict(self.fixed_params),
                    objective_cfg=asdict(self.objective_cfg),
                    constraint_cfg=asdict(self.constraint_cfg),
                    limit_cfg=asdict(self.limit_cfg),
                    decision=item[1],
                )
                for item in uncached_items
            ]

            outcomes = run_batch(tasks, _evaluate_task_worker, max_workers=max_workers, backend="process")
            key_by_idx = {item[0]: item[2] for item in uncached_items}
            for outcome in outcomes:
                idx = outcome.item.request_index
                key = key_by_idx[idx]
                if outcome.ok and outcome.result is not None:
                    res = outcome.result
                else:
                    res = EvaluationResult(
                        objective=float("inf"),
                        objectives={"material_cost": float("inf"), "total_violation": float("inf")},
                        feasible=False,
                        constraints={"worker_failed": 1.0},
                        metrics={"error": outcome.error or "Unknown worker error"},
                    )
                self._cache[key] = res
                results[idx] = res

        return [r for r in results if r is not None]

    def evaluate_objectives(self, x: dict[str, Any]) -> dict[str, float]:
        return dict(self.evaluate(x).objectives)

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


def _evaluate_task_worker(task: BatchEvaluateTask) -> EvaluationResult:
    cfg = DatasetGenerationConfig(**task.analysis_cfg)
    objective_cfg = ShearWallObjectiveConfig(**task.objective_cfg)
    constraint_cfg = ShearWallConstraintConfig(**task.constraint_cfg)
    limit_cfg = ShearWallLimitConfig(**task.limit_cfg)
    input_data, geom_scale = load_and_scale_input(
        json_path=Path(task.layout_path),
        input_unit_scale_to_m=cfg.input_unit_scale_to_m,
        enable_auto_scale=cfg.enable_auto_scale,
        low=cfg.scale_low,
        high=cfg.scale_high,
        seed=cfg.scale_seed,
        manual_factor=cfg.manual_scale_factor,
    )
    return _evaluate_decision(
        input_data=input_data,
        geom_scale=geom_scale,
        decision=task.decision,
        fixed_params=task.fixed_params,
        analysis_cfg=cfg,
        objective_cfg=objective_cfg,
        constraint_cfg=constraint_cfg,
        limit_cfg=limit_cfg,
    )


def _evaluate_decision(
    *,
    input_data,
    geom_scale: float,
    decision: dict[str, Any],
    fixed_params: dict[str, Any],
    analysis_cfg: DatasetGenerationConfig,
    objective_cfg: ShearWallObjectiveConfig,
    constraint_cfg: ShearWallConstraintConfig,
    limit_cfg: ShearWallLimitConfig,
) -> EvaluationResult:
    merged = {**fixed_params, **decision}
    merged.setdefault("conc_mid", merged["conc_bot"])
    merged.setdefault("conc_top", merged["conc_bot"])
    params = ParametricModelParams(
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
    analysis_result = analyze_parametric_model(input_data, params, analysis_cfg)

    total_cost = (
        objective_cfg.concrete_price_per_kg * analysis_result.material_concrete_kg
        + objective_cfg.steel_price_per_kg * analysis_result.material_steel_kg
    )

    def _safe_metric(value: Any, fallback: float) -> float:
        if value is None:
            return fallback
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    metric_values = {
        "torsion_ratio": _safe_metric(analysis_result.torsion_ratio, float("inf")),
        "max_drift_ratio": _safe_metric(analysis_result.max_drift_ratio, float("inf")),
        "period_ratio": _safe_metric(analysis_result.period_ratio, float("inf")),
        "min_shear_weight_ratio": _safe_metric(analysis_result.min_shear_weight_ratio, 0.0),
        "min_stiffness_ratio": _safe_metric(analysis_result.min_stiffness_ratio, 0.0),
    }

    limits = {
        "torsion_ratio": float(limit_cfg.max_torsion_ratio),
        "max_drift_ratio": float(limit_cfg.max_drift_ratio),
        "period_ratio": float(limit_cfg.max_period_ratio),
        "min_shear_weight_ratio": float(limit_cfg.min_shear_weight_ratio),
        "min_stiffness_ratio": float(limit_cfg.min_stiffness_ratio),
    }
    violations: list[float] = []

    for key in ["torsion_ratio", "max_drift_ratio", "period_ratio"]:
        actual = metric_values[key]
        limit = limits[key]
        violations.append(max(0.0, (actual - limit) / limit))

    for key in ["min_shear_weight_ratio", "min_stiffness_ratio"]:
        actual = metric_values[key]
        limit = limits[key]
        violations.append(max(0.0, (limit - actual) / limit))

    total_violation = sum(violations)

    constraints: dict[str, float] = {
        "not_converged": 0.0 if analysis_result.converged else 1.0,
        "analysis_unfeasible": (
            0.0 if (analysis_result.feasible or not constraint_cfg.require_analysis_feasible) else 1.0
        ),
        "design_failed": (
            0.0 if (analysis_result.design_passed or not constraint_cfg.require_design_passed) else 1.0
        ),
    }
    feasible = all(v <= 0.0 for v in constraints.values())

    objective = total_cost * (1.0 + total_violation)
    if not feasible:
        objective += objective_cfg.infeasible_penalty * sum(constraints.values())

    objectives = {
        "material_cost": total_cost,
        "total_violation": total_violation,
    }

    return EvaluationResult(
        objective=objective,
        objectives=objectives,
        feasible=feasible,
        constraints=constraints,
        metrics={
            "geom_scale": geom_scale,
            **asdict(params),
            **asdict(analysis_result),
            "normalized_violations": {
                "torsion_ratio": violations[0],
                "max_drift_ratio": violations[1],
                "period_ratio": violations[2],
                "min_shear_weight_ratio": violations[3],
                "min_stiffness_ratio": violations[4],
            },
            **objectives,
            "objective": objective,
        },
    )
