import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import EvaluationResult, OptimizationResult
from ..core.optimizer import Optimizer
from ..local_calibration import OnlineLocalCalibrationConfig, OnlineLocalCalibrator
from ..surrogate_cost import GNNMaterialCostEstimator, SurrogateCostPrediction, SurrogateCostPreselectionConfig
from ..surrogate_screening import GNNFeasibilityScreener, ScreeningDecision, SurrogateScreeningConfig


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 24
    generations: int = 20
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elite_size: int = 2
    tournament_size: int = 3
    max_workers: int | None = None
    verbose: bool = False
    log_every: int = 1
    seed: int = 42
    surrogate_screening: SurrogateScreeningConfig | None = None
    cost_preselection: SurrogateCostPreselectionConfig | None = None
    local_calibration: OnlineLocalCalibrationConfig | None = None


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, problem, config: GeneticAlgorithmConfig | None = None):
        super().__init__(problem)
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.seed)
        self.screener = self._build_screener()
        self.cost_estimator = self._build_cost_estimator()
        self.local_calibrator = OnlineLocalCalibrator(
            self.config.local_calibration or OnlineLocalCalibrationConfig(enabled=False)
        )

    def optimize(self) -> OptimizationResult:
        print("[GA] Starting optimization")
        pop = [self.problem.repair(self.problem.sample()) for _ in range(self.config.population_size)]
        scored = self._evaluate_population(pop)
        history: list[dict[str, Any]] = []
        global_best_item: tuple[dict[str, Any], EvaluationResult] | None = None

        for gen in range(1, self.config.generations + 1):
            raw_scored = list(scored)
            scored.sort(key=lambda item: self._rank_key(item[1]))
            best_x, best_res = scored[0]
            if global_best_item is None or self._rank_key(best_res) < self._rank_key(global_best_item[1]):
                global_best_item = (dict(best_x), best_res)
            global_best_res = global_best_item[1]
            feasible_count = sum(1 for _, res in raw_scored if res.feasible)
            screened_count = sum(
                1 for _, res in raw_scored if res.constraints.get("surrogate_screen_reject", 0.0) > 0.0
            )
            cost_preselect_skipped_count = sum(
                1 for _, res in raw_scored if bool(res.metrics.get("cost_preselect_skip", False))
            )
            cost_preselect_selected_count = sum(
                1 for _, res in raw_scored if bool(res.metrics.get("cost_preselect_selected", False))
            )
            feasible_ratio = feasible_count / max(1, len(raw_scored))
            history.append(
                {
                    "generation": gen,
                    "best_objective": best_res.objective,
                    "best_feasible": best_res.feasible,
                    "global_best_objective": global_best_res.objective,
                    "global_best_feasible": global_best_res.feasible,
                    "feasible_count": feasible_count,
                    "feasible_ratio": feasible_ratio,
                    "screened_count": screened_count,
                    "screened_ratio": screened_count / max(1, len(raw_scored)),
                    "cost_preselect_selected_count": cost_preselect_selected_count,
                    "cost_preselect_selected_ratio": cost_preselect_selected_count / max(1, len(raw_scored)),
                    "cost_preselect_skipped_count": cost_preselect_skipped_count,
                    "cost_preselect_skipped_ratio": cost_preselect_skipped_count / max(1, len(raw_scored)),
                    "fea_evaluation_count": int(getattr(self.problem, "fea_evaluation_count", -1)),
                    **self.local_calibrator.summary(),
                    "population": [
                        {
                            "objective": float(res.objective),
                            "feasible": bool(res.feasible),
                            "material_cost": float(res.objectives.get("material_cost", float("nan"))),
                            "surrogate_screen_reject": bool(
                                res.constraints.get("surrogate_screen_reject", 0.0) > 0.0
                            ),
                            "cost_preselect_skip": bool(res.metrics.get("cost_preselect_skip", False)),
                            "cost_preselect_selected": bool(res.metrics.get("cost_preselect_selected", False)),
                            "surrogate_cost_score": float(res.metrics.get("surrogate_cost_score", float("nan"))),
                            "surrogate_material_cost": float(res.metrics.get("surrogate_material_cost", float("nan"))),
                            "surrogate_concrete_cost": float(res.metrics.get("surrogate_concrete_cost", float("nan"))),
                            "surrogate_steel_cost": float(res.metrics.get("surrogate_steel_cost", float("nan"))),
                            "surrogate_concrete_kg": float(res.metrics.get("surrogate_concrete_kg", float("nan"))),
                            "surrogate_steel_kg": float(res.metrics.get("surrogate_steel_kg", float("nan"))),
                            "surrogate_final_pass_prob": float(
                                res.metrics.get("surrogate_final_pass_prob", float("nan"))
                            ),
                            "surrogate_infeasible_risk": float(
                                res.metrics.get("surrogate_infeasible_risk", float("nan"))
                            ),
                        }
                        for _, res in raw_scored
                    ],
                    "best_constraints": dict(best_res.constraints),
                    "worst_objectives": scored[-1][1].objectives,
                }
            )

            if self.config.verbose and (
                gen == 1 or gen == self.config.generations or gen % max(1, self.config.log_every) == 0
            ):
                print(
                    f"[GA] gen={gen:03d}/{self.config.generations} "
                    f"best={best_res.objective:.6g} "
                    f"global_best={global_best_res.objective:.6g} "
                    f"best_feasible={best_res.feasible} "
                    f"global_best_feasible={global_best_res.feasible} "
                    f"feasible_ratio={feasible_ratio:.1%} "
                    f"screened_ratio={screened_count / max(1, len(raw_scored)):.1%} "
                    f"cost_selected_ratio={cost_preselect_selected_count / max(1, len(raw_scored)):.1%} "
                    f"cost_skipped_ratio={cost_preselect_skipped_count / max(1, len(raw_scored)):.1%} "
                    f"fea_count={int(getattr(self.problem, 'fea_evaluation_count', -1))}"
                )

            elites = [x for x, _ in scored[: self.config.elite_size]]
            next_pop = [dict(x) for x in elites]

            while len(next_pop) < self.config.population_size:
                p1 = self._tournament_pick(scored)
                p2 = self._tournament_pick(scored)
                c1, c2 = self._crossover(p1, p2)
                next_pop.append(self._mutate(c1))
                if len(next_pop) < self.config.population_size:
                    next_pop.append(self._mutate(c2))

            scored = self._evaluate_population(next_pop)

        if global_best_item is None:
            scored.sort(key=lambda item: self._rank_key(item[1]))
            global_best_item = scored[0]
        best_x, best_res = global_best_item
        if self._is_synthetic_result(best_res):
            best_res = self.problem.evaluate(best_x)
        if history:
            history[-1]["final_fea_evaluation_count"] = int(getattr(self.problem, "fea_evaluation_count", -1))
        return OptimizationResult(
            best_solution=best_x,
            best_objective=best_res.objective,
            best_objectives=best_res.objectives,
            best_feasible=best_res.feasible,
            best_constraints=best_res.constraints,
            history=history,
        )

    @staticmethod
    def _rank_key(res: EvaluationResult) -> tuple[bool, float]:
        return (not res.feasible, float(res.objective))

    def _evaluate_population(
        self, pop: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], EvaluationResult]]:
        repaired = [self.problem.repair(x) for x in pop]
        if self.screener is not None or self.cost_estimator is not None:
            return self._evaluate_population_with_surrogates(repaired)
        results = self._real_evaluate_many(repaired)
        return list(zip(repaired, results))

    def _evaluate_population_with_surrogates(
        self, repaired: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], EvaluationResult]]:
        n = len(repaired)
        results: list[EvaluationResult | None] = [None] * n
        raw_screening = self.screener.screen(repaired) if self.screener is not None else [None] * n
        screening = self._calibrate_screening(raw_screening)

        candidate_indices = []
        for i, decision in enumerate(screening):
            if decision is not None and decision.reject:
                results[i] = self._screened_result(decision)
            else:
                candidate_indices.append(i)

        cost_preds: dict[int, SurrogateCostPrediction] = {}
        global_cost_preds: dict[int, SurrogateCostPrediction] = {}
        if self.cost_estimator is not None and candidate_indices:
            items = [repaired[i] for i in candidate_indices]
            probs = [
                None if screening[i] is None else float(screening[i].probability)
                for i in candidate_indices
            ]
            preds = self.cost_estimator.predict(items, feasible_probabilities=probs)
            global_cost_preds = dict(zip(candidate_indices, preds))
            cost_preds = {
                idx: self._calibrate_cost_prediction(repaired[idx], pred)
                for idx, pred in global_cost_preds.items()
            }
            selected_indices = self._select_cost_candidates(candidate_indices, cost_preds)
        else:
            selected_indices = list(candidate_indices)

        selected_items = [repaired[i] for i in selected_indices]
        eval_results = self._real_evaluate_many(selected_items)

        for idx, res in zip(selected_indices, eval_results):
            metrics = dict(res.metrics)
            pred = cost_preds.get(idx)
            if pred is not None:
                metrics.update(self._cost_prediction_metrics(pred, selected=True))
            self._update_local_calibration(
                repaired[idx],
                res,
                raw_screening[idx],
                global_cost_preds.get(idx),
            )
            results[idx] = EvaluationResult(
                objective=res.objective,
                objectives=res.objectives,
                feasible=res.feasible,
                constraints=res.constraints,
                metrics=metrics,
            )

        for i in range(n):
            if results[i] is None:
                pred = cost_preds.get(i)
                results[i] = self._cost_preselect_skipped_result(pred)
        return [(x, res) for x, res in zip(repaired, results) if res is not None]

    def _calibrate_screening(
        self, raw: list[ScreeningDecision | None]
    ) -> list[ScreeningDecision | None]:
        out = []
        for decision in raw:
            if decision is None:
                out.append(None)
                continue
            cal = self.local_calibrator.calibrate_feasibility(decision.probability, decision.threshold)
            out.append(
                ScreeningDecision(
                    reject=cal.reject,
                    probability=cal.probability,
                    threshold=cal.threshold,
                )
            )
        return out

    def _calibrate_cost_prediction(
        self,
        decision: dict[str, Any],
        pred: SurrogateCostPrediction,
    ) -> SurrogateCostPrediction:
        steel_kg, _, _ = self.local_calibrator.calibrate_steel(decision, pred.steel_kg)
        steel_cost = self.problem.objective_cfg.steel_price_per_kg * steel_kg
        material = pred.concrete_cost + steel_cost
        cfg = self.config.cost_preselection or SurrogateCostPreselectionConfig(enabled=False)
        score = material + float(cfg.feasibility_penalty_cost) * float(pred.infeasible_risk)
        return SurrogateCostPrediction(
            score=float(score),
            steel_kg=float(steel_kg),
            concrete_kg=float(pred.concrete_kg),
            concrete_cost=float(pred.concrete_cost),
            steel_cost=float(steel_cost),
            material_cost=float(material),
            feasible_probability=pred.feasible_probability,
            infeasible_risk=float(pred.infeasible_risk),
        )

    def _update_local_calibration(
        self,
        decision: dict[str, Any],
        result: EvaluationResult,
        screening: ScreeningDecision | None,
        pred: SurrogateCostPrediction | None,
    ) -> None:
        if not self.local_calibrator.cfg.enabled:
            return
        self.local_calibrator.update(
            decision,
            result,
            global_probability=None if screening is None else float(screening.probability),
            global_steel_kg=None if pred is None else float(pred.steel_kg),
        )

    def _select_cost_candidates(
        self,
        indices: list[int],
        preds: dict[int, SurrogateCostPrediction],
    ) -> list[int]:
        cfg = self.config.cost_preselection or SurrogateCostPreselectionConfig(enabled=False)
        n = len(indices)
        eval_count = min(n, max(int(cfg.min_eval_count), int(round(n * cfg.eval_ratio))))
        return sorted(sorted(indices, key=lambda i: preds[i].score)[:eval_count])

    def _screened_result(self, decision: ScreeningDecision) -> EvaluationResult:
        cfg = self.config.surrogate_screening or SurrogateScreeningConfig(enabled=False)
        objective = float(cfg.rejected_objective)
        return EvaluationResult(
            objective=objective,
            objectives={"material_cost": objective, "total_violation": 1.0},
            feasible=False,
            constraints={"surrogate_screen_reject": 1.0},
            metrics={
                "surrogate_screen_reject": True,
                "surrogate_final_pass_prob": float(decision.probability),
                "surrogate_screening_threshold": float(decision.threshold),
            },
        )

    def _cost_preselect_skipped_result(self, pred: SurrogateCostPrediction | None) -> EvaluationResult:
        cfg = self.config.cost_preselection or SurrogateCostPreselectionConfig(enabled=False)
        objective = float(cfg.skipped_objective)
        metrics: dict[str, float | bool] = {"cost_preselect_skip": True}
        if pred is not None:
            metrics.update(self._cost_prediction_metrics(pred, selected=False))
        return EvaluationResult(
            objective=objective,
            objectives={"material_cost": objective, "total_violation": 1.0},
            feasible=False,
            constraints={"cost_preselect_skip": 1.0},
            metrics=metrics,
        )

    @staticmethod
    def _cost_prediction_metrics(pred: SurrogateCostPrediction, *, selected: bool) -> dict[str, float | bool]:
        return {
            "surrogate_cost_score": float(pred.score),
            "surrogate_material_cost": float(pred.material_cost),
            "surrogate_concrete_cost": float(pred.concrete_cost),
            "surrogate_steel_cost": float(pred.steel_cost),
            "surrogate_concrete_kg": float(pred.concrete_kg),
            "surrogate_steel_kg": float(pred.steel_kg),
            "surrogate_final_pass_prob": (
                float("nan") if pred.feasible_probability is None else float(pred.feasible_probability)
            ),
            "surrogate_infeasible_risk": float(pred.infeasible_risk),
            "cost_preselect_selected": bool(selected),
        }

    @staticmethod
    def _is_synthetic_result(res: EvaluationResult) -> bool:
        return bool(
            res.metrics.get("cost_preselect_skip", False)
            or res.metrics.get("surrogate_screen_reject", False)
        )

    def _real_evaluate_many(self, items: list[dict[str, Any]]) -> list[EvaluationResult]:
        if not items:
            return []
        if hasattr(self.problem, "evaluate_many"):
            return self.problem.evaluate_many(items, max_workers=self.config.max_workers)
        return [self.problem.evaluate(x) for x in items]

    def _build_screener(self) -> GNNFeasibilityScreener | None:
        cfg = self.config.surrogate_screening
        if cfg is None or not cfg.enabled:
            return None
        if (
            not hasattr(self.problem, "layout_path")
            or not hasattr(self.problem, "fixed_params")
        ):
            raise ValueError("Surrogate screening requires problem.layout_path and problem.fixed_params.")
        return GNNFeasibilityScreener(
            cfg,
            layout_id=self.problem.layout_path.stem,
            fixed_params=self.problem.fixed_params,
        )

    def _build_cost_estimator(self) -> GNNMaterialCostEstimator | None:
        cfg = self.config.cost_preselection
        if cfg is None or not cfg.enabled:
            return None
        if (
            not hasattr(self.problem, "layout_path")
            or not hasattr(self.problem, "fixed_params")
            or not hasattr(self.problem, "objective_cfg")
        ):
            raise ValueError("Cost preselection requires problem.layout_path, fixed_params and objective_cfg.")
        return GNNMaterialCostEstimator(
            cfg,
            layout_id=self.problem.layout_path.stem,
            fixed_params=self.problem.fixed_params,
            steel_price_per_kg=self.problem.objective_cfg.steel_price_per_kg,
        )

    def _tournament_pick(self, scored: list[tuple[dict[str, Any], EvaluationResult]]) -> dict[str, Any]:
        pool = self.rng.sample(scored, k=min(self.config.tournament_size, len(scored)))
        pool.sort(key=lambda item: (not item[1].feasible, item[1].objective))
        return dict(pool[0][0])

    def _crossover(self, p1: dict[str, Any], p2: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.rng.random() > self.config.crossover_rate:
            return dict(p1), dict(p2)

        c1: dict[str, Any] = {}
        c2: dict[str, Any] = {}
        for spec in self.problem.variables:
            if self.rng.random() < 0.5:
                c1[spec.name] = p1[spec.name]
                c2[spec.name] = p2[spec.name]
            else:
                c1[spec.name] = p2[spec.name]
                c2[spec.name] = p1[spec.name]
        return self.problem.repair(c1), self.problem.repair(c2)

    def _mutate(self, x: dict[str, Any]) -> dict[str, Any]:
        out = dict(x)
        for spec in self.problem.variables:
            if self.rng.random() < self.config.mutation_rate:
                out[spec.name] = self.rng.choice(spec.domain)
        return self.problem.repair(out)
