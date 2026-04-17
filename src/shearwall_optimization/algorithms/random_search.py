import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import OptimizationResult
from ..core.optimizer import Optimizer


@dataclass(frozen=True)
class RandomSearchConfig:
    n_trials: int = 200
    seed: int = 42


class RandomSearchOptimizer(Optimizer):
    def __init__(self, problem, config: RandomSearchConfig | None = None):
        super().__init__(problem)
        self.config = config or RandomSearchConfig()
        self.rng = random.Random(self.config.seed)

    def optimize(self) -> OptimizationResult:
        best_x: dict[str, Any] | None = None
        best_res = None
        history: list[dict[str, Any]] = []

        for i in range(1, self.config.n_trials + 1):
            x = self.problem.repair(self.problem.sample())
            res = self.problem.evaluate(x)
            if best_res is None or res.objective < best_res.objective:
                best_x = dict(x)
                best_res = res
            history.append(
                {
                    "trial": i,
                    "objective": res.objective,
                    "feasible": res.feasible,
                    "best_objective": best_res.objective,
                }
            )

        assert best_x is not None
        assert best_res is not None
        return OptimizationResult(
            best_solution=best_x,
            best_objective=best_res.objective,
            best_feasible=best_res.feasible,
            best_constraints=best_res.constraints,
            history=history,
        )
