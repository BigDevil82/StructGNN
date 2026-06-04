import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import EvaluationResult, OptimizationResult
from ..core.optimizer import Optimizer
from ..local_calibration import OnlineLocalCalibrationConfig
from ..surrogate_candidate import SurrogateCandidateEvaluator, SurrogateEvaluationConfig
from ..surrogate_cost import SurrogateCostPreselectionConfig
from ..surrogate_screening import SurrogateScreeningConfig


@dataclass(frozen=True)
class RandomSearchConfig:
    n_trials: int = 200
    batch_size: int = 24
    max_workers: int | None = None
    seed: int = 42
    surrogate_screening: SurrogateScreeningConfig | None = None
    cost_preselection: SurrogateCostPreselectionConfig | None = None
    local_calibration: OnlineLocalCalibrationConfig | None = None


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem, config: RandomSearchConfig | None = None):
        super().__init__(problem)
        self.config = config or RandomSearchConfig()
        self.rng = random.Random(self.config.seed)
        self.evaluator = SurrogateCandidateEvaluator(
            problem,
            SurrogateEvaluationConfig(
                max_workers=self.config.max_workers,
                surrogate_screening=self.config.surrogate_screening,
                cost_preselection=self.config.cost_preselection,
                local_calibration=self.config.local_calibration,
            ),
        )

    def optimize(self) -> OptimizationResult:
        print("[Random] Starting optimization")
        best_x: dict[str, Any] | None = None
        best_res: EvaluationResult | None = None
        history: list[dict[str, Any]] = []
        remaining = int(self.config.n_trials)
        batch_size = max(1, int(self.config.batch_size))
        trial_count = 0
        batch_id = 0

        while remaining > 0:
            batch_id += 1
            n = min(batch_size, remaining)
            candidates = [self._sample() for _ in range(n)]
            evaluated = self.evaluator.evaluate(candidates)
            results = [res for _, res in evaluated]
            for x, res in evaluated:
                trial_count += 1
                if best_res is None or self._rank_key(res) < self._rank_key(best_res):
                    best_x = dict(x)
                    best_res = res
            counts = self.evaluator.population_counts(results)
            history.append(
                {
                    "iteration": batch_id,
                    "trial": trial_count,
                    "best_objective": best_res.objective,
                    "best_feasible": best_res.feasible,
                    **counts,
                    "fea_evaluation_count": int(getattr(self.problem, "fea_evaluation_count", -1)),
                    **self.evaluator.local_summary(),
                    "population": [self.evaluator.population_item(res) for res in results],
                    "best_constraints": dict(best_res.constraints),
                }
            )
            remaining -= n

        assert best_x is not None
        assert best_res is not None
        if self.evaluator.is_synthetic_result(best_res):
            best_x = self.problem.repair(best_x)
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

    def _sample(self) -> dict[str, Any]:
        return self.problem.repair({spec.name: self.rng.choice(spec.domain) for spec in self.problem.variables})

    @staticmethod
    def _rank_key(res: EvaluationResult) -> tuple[bool, float]:
        return SurrogateCandidateEvaluator.rank_key(res)
