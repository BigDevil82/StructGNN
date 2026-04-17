import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import EvaluationResult, OptimizationResult
from ..core.optimizer import Optimizer


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    population_size: int = 24
    generations: int = 20
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elite_size: int = 2
    tournament_size: int = 3
    max_workers: int | None = None
    seed: int = 42


class GeneticAlgorithmOptimizer(Optimizer):
    def __init__(self, problem, config: GeneticAlgorithmConfig | None = None):
        super().__init__(problem)
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.seed)

    def optimize(self) -> OptimizationResult:
        pop = [self.problem.repair(self.problem.sample()) for _ in range(self.config.population_size)]
        scored = self._evaluate_population(pop)
        history: list[dict[str, Any]] = []

        for gen in range(1, self.config.generations + 1):
            scored.sort(key=lambda item: item[1].objective)
            best_x, best_res = scored[0]
            history.append(
                {
                    "generation": gen,
                    "best_objective": best_res.objective,
                    "best_feasible": best_res.feasible,
                    "best_constraints": dict(best_res.constraints),
                }
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

        scored.sort(key=lambda item: item[1].objective)
        best_x, best_res = scored[0]
        return OptimizationResult(
            best_solution=best_x,
            best_objective=best_res.objective,
            best_feasible=best_res.feasible,
            best_constraints=best_res.constraints,
            history=history,
        )

    def _evaluate_population(
        self, pop: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], EvaluationResult]]:
        repaired = [self.problem.repair(x) for x in pop]
        if hasattr(self.problem, "evaluate_many"):
            results = self.problem.evaluate_many(repaired, max_workers=self.config.max_workers)
            return list(zip(repaired, results))
        return [(x, self.problem.evaluate(x)) for x in repaired]

    def _tournament_pick(self, scored: list[tuple[dict[str, Any], EvaluationResult]]) -> dict[str, Any]:
        pool = self.rng.sample(scored, k=min(self.config.tournament_size, len(scored)))
        pool.sort(key=lambda item: item[1].objective)
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
