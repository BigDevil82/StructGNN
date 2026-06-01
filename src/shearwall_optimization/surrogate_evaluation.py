from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.shearwall_optimization.core.contracts import EvaluationResult
from src.shearwall_optimization.problems.shearwall_problem import ShearWallObjectiveConfig

from .surrogate_material import SteelMaterialPrediction, SteelMaterialPredictor
from .surrogate_screening import GNNFeasibilityScreener, ScreeningDecision, SurrogateScreeningConfig


@dataclass(frozen=True)
class SurrogateAcceptanceConfig:
    enabled: bool = False
    steel_artifact_dir: str = r"data\parametric\surrogate_dataset\steel_quantile_lightgbm"
    pass_probability_threshold: float = 0.99
    max_steel_rel_upper_gap: float = 0.75
    audit_rate: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class SurrogateEvaluationConfig:
    screening: SurrogateScreeningConfig | None = None
    acceptance: SurrogateAcceptanceConfig | None = None

    @property
    def enabled(self) -> bool:
        return bool(
            (self.screening is not None and self.screening.enabled)
            or (self.acceptance is not None and self.acceptance.enabled)
        )


class SurrogateEvaluationAccelerator:
    def __init__(
        self,
        cfg: SurrogateEvaluationConfig,
        *,
        layout_path: str | Path,
        fixed_params: dict[str, Any],
        objective_cfg: ShearWallObjectiveConfig,
    ):
        self.cfg = cfg
        self.objective_cfg = objective_cfg
        self.acceptance_cfg = cfg.acceptance or SurrogateAcceptanceConfig(enabled=False)
        self.rng = random.Random(self.acceptance_cfg.seed)

        screening_cfg = cfg.screening or SurrogateScreeningConfig(enabled=False)
        if not screening_cfg.enabled and self.acceptance_cfg.enabled:
            screening_cfg = SurrogateScreeningConfig(
                enabled=True,
                artifact_path=screening_cfg.artifact_path,
                graph_cache_dir=screening_cfg.graph_cache_dir,
                layout_features_path=screening_cfg.layout_features_path,
                screening_threshold=screening_cfg.screening_threshold,
                batch_size=screening_cfg.batch_size,
                num_workers=screening_cfg.num_workers,
                rejected_objective=screening_cfg.rejected_objective,
            )

        self.screening_cfg = screening_cfg
        self.screener = GNNFeasibilityScreener(
            screening_cfg,
            layout_id=Path(layout_path).stem,
            fixed_params=fixed_params,
        )
        self.material_predictor = (
            SteelMaterialPredictor(self.acceptance_cfg.steel_artifact_dir)
            if self.acceptance_cfg.enabled
            else None
        )

    def evaluate_many(
        self,
        xs: list[dict[str, Any]],
        real_evaluate_many: Callable[[list[dict[str, Any]]], list[EvaluationResult]],
    ) -> list[EvaluationResult]:
        decisions = self.screener.screen(xs)
        frames = self.screener.build_frame(xs)
        material = (
            self.material_predictor.predict(frames)
            if self.material_predictor is not None
            else [None] * len(xs)
        )

        results: list[EvaluationResult | None] = [None] * len(xs)
        real_indices: list[int] = []
        real_items: list[dict[str, Any]] = []

        for i, (decision, mat) in enumerate(zip(decisions, material)):
            if self.screening_cfg.enabled and decision.reject:
                results[i] = self._screened_result(decision)
                continue

            if self._can_accept(decision, mat) and not self._audit():
                results[i] = self._accepted_result(decision, mat)
                continue

            real_indices.append(i)
            real_items.append(xs[i])

        if real_items:
            real_results = real_evaluate_many(real_items)
            for idx, res in zip(real_indices, real_results):
                results[idx] = res

        return [res for res in results if res is not None]

    def _can_accept(self, decision: ScreeningDecision, mat: SteelMaterialPrediction | None) -> bool:
        if not self.acceptance_cfg.enabled or mat is None:
            return False
        return (
            decision.probability >= self.acceptance_cfg.pass_probability_threshold
            and mat.steel_rel_upper_gap <= self.acceptance_cfg.max_steel_rel_upper_gap
        )

    def _audit(self) -> bool:
        return self.acceptance_cfg.audit_rate > 0.0 and self.rng.random() < self.acceptance_cfg.audit_rate

    def _screened_result(self, decision: ScreeningDecision) -> EvaluationResult:
        objective = float(self.screening_cfg.rejected_objective)
        return EvaluationResult(
            objective=objective,
            objectives={"material_cost": objective, "total_violation": 1.0},
            feasible=False,
            constraints={"surrogate_screen_reject": 1.0},
            metrics={
                "surrogate_screen_reject": True,
                "surrogate_final_pass_prob": decision.probability,
                "surrogate_screening_threshold": decision.threshold,
            },
        )

    def _accepted_result(self, decision: ScreeningDecision, mat: SteelMaterialPrediction) -> EvaluationResult:
        concrete_cost = _concrete_cost(mat.concrete_kg)
        steel_cost = self.objective_cfg.steel_price_per_kg * mat.steel_upper_kg
        material_cost = concrete_cost + steel_cost
        return EvaluationResult(
            objective=float(material_cost),
            objectives={
                "material_cost": float(material_cost),
                "total_violation": 0.0,
                "surrogate_concrete_cost": float(concrete_cost),
                "surrogate_steel_cost_upper": float(steel_cost),
            },
            feasible=True,
            constraints={},
            metrics={
                "surrogate_material_accept": True,
                "surrogate_final_pass_prob": decision.probability,
                "surrogate_steel_mean_kg": mat.steel_mean_kg,
                "surrogate_steel_upper_kg": mat.steel_upper_kg,
                "surrogate_steel_rel_upper_gap": mat.steel_rel_upper_gap,
                "surrogate_concrete_kg": mat.concrete_kg,
                "objective": float(material_cost),
            },
        )


def _concrete_cost(concrete_kg: float) -> float:
    # Use the highest supported concrete unit price as a conservative bound.
    return (620.0 / 2400.0) * float(concrete_kg)
