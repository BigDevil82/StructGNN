from dataclasses import dataclass
from typing import Any

import optuna

from ..core.contracts import OptimizationResult
from ..core.optimizer import Optimizer


@dataclass(frozen=True)
class NSGA2Config:
    n_trials: int = 80
    population_size: int = 24
    seed: int = 42


class NSGA2Optimizer(Optimizer):
    def __init__(self, problem, config: NSGA2Config | None = None):
        super().__init__(problem)
        self.config = config or NSGA2Config()

    def optimize(self) -> OptimizationResult:
        sampler = optuna.samplers.NSGAIISampler(
            seed=self.config.seed,
            population_size=self.config.population_size,
        )
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)

        def objective(trial: optuna.trial.Trial):
            x = {}
            for spec in self.problem.variables:
                x[spec.name] = trial.suggest_categorical(spec.name, spec.domain)
            res = self.problem.evaluate(x)
            material = float(res.objectives.get("material_cost", res.objective))
            margin_penalty = float(res.objectives.get("margin_penalty", 0.0))
            trial.set_user_attr("scalar_objective", float(res.objective))
            trial.set_user_attr("feasible", bool(res.feasible))
            trial.set_user_attr("constraints", dict(res.constraints))
            trial.set_user_attr("objectives", dict(res.objectives))
            return material, margin_penalty

        study.optimize(objective, n_trials=self.config.n_trials, n_jobs=1)

        completed = [tr for tr in study.trials if tr.values is not None]
        if not completed:
            raise RuntimeError("No completed NSGA-II trials.")

        best_trial = min(completed, key=lambda tr: float(tr.user_attrs.get("scalar_objective", float("inf"))))
        best_x: dict[str, Any] = dict(best_trial.params)
        best_res = self.problem.evaluate(best_x)

        history: list[dict[str, Any]] = []
        pareto_trial_ids = {tr.number for tr in study.best_trials}
        for tr in completed:
            vals = tr.values or [float("inf"), float("inf")]
            history.append(
                {
                    "trial": tr.number,
                    "material_cost": float(vals[0]),
                    "margin_penalty": float(vals[1]),
                    "scalar_objective": float(tr.user_attrs.get("scalar_objective", float("inf"))),
                    "feasible": bool(tr.user_attrs.get("feasible", False)),
                    "constraints": dict(tr.user_attrs.get("constraints", {})),
                    "objectives": dict(tr.user_attrs.get("objectives", {})),
                    "pareto": tr.number in pareto_trial_ids,
                    "params": dict(tr.params),
                }
            )

        return OptimizationResult(
            best_solution=best_x,
            best_objective=best_res.objective,
            best_objectives=best_res.objectives,
            best_feasible=best_res.feasible,
            best_constraints=best_res.constraints,
            history=history,
        )
