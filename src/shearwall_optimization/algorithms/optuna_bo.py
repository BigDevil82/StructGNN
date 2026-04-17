from dataclasses import dataclass
from typing import Any

import optuna

from ..core.contracts import OptimizationResult
from ..core.optimizer import Optimizer


@dataclass(frozen=True)
class OptunaBayesConfig:
    n_trials: int = 50
    n_startup_trials: int = 10
    seed: int = 42
    show_progress_bar: bool = False


class OptunaBayesOptimizer(Optimizer):
    def __init__(self, problem, config: OptunaBayesConfig | None = None):
        super().__init__(problem)
        self.config = config or OptunaBayesConfig()

    def optimize(self) -> OptimizationResult:
        sampler = optuna.samplers.TPESampler(
            seed=self.config.seed,
            n_startup_trials=self.config.n_startup_trials,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)

        def objective(trial: optuna.trial.Trial) -> float:
            x = {}
            for spec in self.problem.variables:
                x[spec.name] = trial.suggest_categorical(spec.name, spec.domain)
            res = self.problem.evaluate(x)
            trial.set_user_attr("feasible", res.feasible)
            trial.set_user_attr("constraints", dict(res.constraints))
            trial.set_user_attr("objectives", dict(res.objectives))
            return float(res.objective)

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            n_jobs=1,
            show_progress_bar=self.config.show_progress_bar,
        )

        best_x: dict[str, Any] = dict(study.best_params)
        best_res = self.problem.evaluate(best_x)

        history: list[dict[str, Any]] = []
        for tr in study.trials:
            if tr.value is None:
                continue
            history.append(
                {
                    "trial": tr.number,
                    "objective": float(tr.value),
                    "params": dict(tr.params),
                    "feasible": bool(tr.user_attrs.get("feasible", False)),
                    "constraints": dict(tr.user_attrs.get("constraints", {})),
                    "objectives": dict(tr.user_attrs.get("objectives", {})),
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
