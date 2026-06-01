import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import EvaluationResult, OptimizationResult
from ..core.optimizer import Optimizer
from ..surrogate_evaluation import (
    SurrogateAcceptanceConfig,
    SurrogateEvaluationAccelerator,
    SurrogateEvaluationConfig,
)
from ..surrogate_screening import SurrogateScreeningConfig
from ..steel_ranking import GNNSteelRanker, SteelRankingConfig


@dataclass(frozen=True)
class RandomPreselectionConfig:
    enabled: bool = False
    eval_ratio: float = 0.4
    min_eval_count: int = 8
    skipped_objective: float = 1.0e12


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
    surrogate_acceptance: SurrogateAcceptanceConfig | None = None
    steel_ranking: SteelRankingConfig | None = None
    random_preselection: RandomPreselectionConfig | None = None


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, problem, config: GeneticAlgorithmConfig | None = None):
        super().__init__(problem)
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.seed)
        self.surrogate = self._build_surrogate()
        self.steel_ranker = self._build_steel_ranker()

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
            surrogate_accepted_count = sum(
                1 for _, res in raw_scored if bool(res.metrics.get("surrogate_material_accept", False))
            )
            steel_rank_skipped_count = sum(
                1 for _, res in raw_scored if bool(res.metrics.get("steel_rank_skip", False))
            )
            random_preselect_skipped_count = sum(
                1 for _, res in raw_scored if bool(res.metrics.get("random_preselect_skip", False))
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
                    "surrogate_accepted_count": surrogate_accepted_count,
                    "surrogate_accepted_ratio": surrogate_accepted_count / max(1, len(raw_scored)),
                    "steel_rank_skipped_count": steel_rank_skipped_count,
                    "steel_rank_skipped_ratio": steel_rank_skipped_count / max(1, len(raw_scored)),
                    "random_preselect_skipped_count": random_preselect_skipped_count,
                    "random_preselect_skipped_ratio": random_preselect_skipped_count / max(1, len(raw_scored)),
                    "fea_evaluation_count": int(getattr(self.problem, "fea_evaluation_count", -1)),
                    "population": [
                        {
                            "objective": float(res.objective),
                            "feasible": bool(res.feasible),
                            "material_cost": float(res.objectives.get("material_cost", float("nan"))),
                            "surrogate_screen_reject": bool(
                                res.constraints.get("surrogate_screen_reject", 0.0) > 0.0
                            ),
                            "surrogate_material_accept": bool(res.metrics.get("surrogate_material_accept", False)),
                            "steel_rank_skip": bool(res.metrics.get("steel_rank_skip", False)),
                            "steel_rank_score": float(res.metrics.get("steel_rank_score", float("nan"))),
                            "random_preselect_skip": bool(res.metrics.get("random_preselect_skip", False)),
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
                    f"surrogate_accepted_ratio={surrogate_accepted_count / max(1, len(raw_scored)):.1%} "
                    f"steel_rank_skipped_ratio={steel_rank_skipped_count / max(1, len(raw_scored)):.1%} "
                    f"random_skip_ratio={random_preselect_skipped_count / max(1, len(raw_scored)):.1%} "
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
        if bool(best_res.metrics.get("surrogate_material_accept", False)):
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
        if self.steel_ranker is not None:
            return self._evaluate_population_with_steel_ranking(repaired)
        if self.config.random_preselection is not None and self.config.random_preselection.enabled:
            return self._evaluate_population_with_random_preselection(repaired)
        if self.surrogate is not None:
            results = self.surrogate.evaluate_many(repaired, self._real_evaluate_many)
            return list(zip(repaired, results))
        results = self._real_evaluate_many(repaired)
        return list(zip(repaired, results))

    def _evaluate_population_with_random_preselection(
        self, repaired: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], EvaluationResult]]:
        cfg = self.config.random_preselection or RandomPreselectionConfig(enabled=False)
        n = len(repaired)
        eval_count = min(n, max(int(cfg.min_eval_count), int(round(n * cfg.eval_ratio))))
        selected_indices = sorted(self.rng.sample(range(n), k=eval_count))
        selected_items = [repaired[i] for i in selected_indices]
        if self.surrogate is not None:
            eval_results = self.surrogate.evaluate_many(selected_items, self._real_evaluate_many)
        else:
            eval_results = self._real_evaluate_many(selected_items)

        results: list[EvaluationResult | None] = [None] * n
        for idx, res in zip(selected_indices, eval_results):
            metrics = dict(res.metrics)
            metrics["random_preselect_selected"] = True
            results[idx] = EvaluationResult(
                objective=res.objective,
                objectives=res.objectives,
                feasible=res.feasible,
                constraints=res.constraints,
                metrics=metrics,
            )

        for i in range(n):
            if results[i] is None:
                results[i] = self._random_preselection_skipped_result(cfg)
        return [(x, res) for x, res in zip(repaired, results) if res is not None]

    def _random_preselection_skipped_result(self, cfg: RandomPreselectionConfig) -> EvaluationResult:
        objective = float(cfg.skipped_objective)
        return EvaluationResult(
            objective=objective,
            objectives={"material_cost": objective, "total_violation": 1.0},
            feasible=False,
            constraints={"random_preselect_skip": 1.0},
            metrics={"random_preselect_skip": True},
        )

    def _evaluate_population_with_steel_ranking(
        self, repaired: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], EvaluationResult]]:
        assert self.steel_ranker is not None
        cfg = self.config.steel_ranking or SteelRankingConfig(enabled=False)
        scores = self.steel_ranker.predict(repaired)
        n = len(repaired)
        eval_count = min(n, max(int(cfg.min_eval_count), int(round(n * cfg.eval_ratio))))
        order = sorted(range(n), key=lambda i: scores[i])
        selected = set(order[:eval_count])

        remaining = [i for i in range(n) if i not in selected]
        random_count = min(len(remaining), int(round(n * cfg.random_ratio)))
        if random_count > 0:
            selected.update(self.rng.sample(remaining, k=random_count))

        selected_indices = sorted(selected)
        selected_items = [repaired[i] for i in selected_indices]
        if self.surrogate is not None:
            eval_results = self.surrogate.evaluate_many(selected_items, self._real_evaluate_many)
        else:
            eval_results = self._real_evaluate_many(selected_items)

        results: list[EvaluationResult | None] = [None] * n
        for idx, res in zip(selected_indices, eval_results):
            metrics = dict(res.metrics)
            metrics["steel_rank_score"] = float(scores[idx])
            metrics["steel_rank_selected"] = True
            results[idx] = EvaluationResult(
                objective=res.objective,
                objectives=res.objectives,
                feasible=res.feasible,
                constraints=res.constraints,
                metrics=metrics,
            )

        for i in range(n):
            if results[i] is None:
                results[i] = self._steel_rank_skipped_result(scores[i], cfg)
        return [(x, res) for x, res in zip(repaired, results) if res is not None]

    def _steel_rank_skipped_result(self, score: float, cfg: SteelRankingConfig) -> EvaluationResult:
        objective = float(cfg.skipped_objective)
        return EvaluationResult(
            objective=objective,
            objectives={"material_cost": objective, "total_violation": 1.0},
            feasible=False,
            constraints={"steel_rank_skip": 1.0},
            metrics={
                "steel_rank_skip": True,
                "steel_rank_score": float(score),
            },
        )

    def _real_evaluate_many(self, items: list[dict[str, Any]]) -> list[EvaluationResult]:
        if not items:
            return []
        if hasattr(self.problem, "evaluate_many"):
            return self.problem.evaluate_many(items, max_workers=self.config.max_workers)
        return [self.problem.evaluate(x) for x in items]

    def _build_surrogate(self) -> SurrogateEvaluationAccelerator | None:
        cfg = SurrogateEvaluationConfig(
            screening=self.config.surrogate_screening,
            acceptance=self.config.surrogate_acceptance,
        )
        if not cfg.enabled:
            return None
        if (
            not hasattr(self.problem, "layout_path")
            or not hasattr(self.problem, "fixed_params")
            or not hasattr(self.problem, "objective_cfg")
        ):
            raise ValueError("Surrogate evaluation requires problem.layout_path, fixed_params and objective_cfg.")
        return SurrogateEvaluationAccelerator(
            cfg,
            layout_path=self.problem.layout_path,
            fixed_params=self.problem.fixed_params,
            objective_cfg=self.problem.objective_cfg,
        )

    def _build_steel_ranker(self) -> GNNSteelRanker | None:
        cfg = self.config.steel_ranking
        if cfg is None or not cfg.enabled:
            return None
        if not hasattr(self.problem, "layout_path") or not hasattr(self.problem, "fixed_params"):
            raise ValueError("Steel ranking requires problem.layout_path and problem.fixed_params.")
        return GNNSteelRanker(
            cfg,
            layout_id=self.problem.layout_path.stem,
            fixed_params=self.problem.fixed_params,
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
