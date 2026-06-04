from dataclasses import dataclass
from typing import Any

import optuna

from ..core.contracts import EvaluationResult, OptimizationResult
from ..core.optimizer import Optimizer
from ..local_calibration import OnlineLocalCalibrationConfig
from ..surrogate_candidate import SurrogateCandidateEvaluator, SurrogateEvaluationConfig
from ..surrogate_cost import SurrogateCostPreselectionConfig
from ..surrogate_screening import SurrogateScreeningConfig


@dataclass(frozen=True)
class OptunaBayesConfig:
    n_trials: int = 50
    n_startup_trials: int = 10
    batch_size: int = 24
    max_workers: int | None = None
    seed: int = 42
    show_progress_bar: bool = False
    surrogate_screening: SurrogateScreeningConfig | None = None
    cost_preselection: SurrogateCostPreselectionConfig | None = None
    local_calibration: OnlineLocalCalibrationConfig | None = None


class OptunaBayesOptimizer(Optimizer):
    def __init__(self, problem, config: OptunaBayesConfig | None = None):
        super().__init__(problem)
        self.config = config or OptunaBayesConfig()
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
        print("[Optuna] Starting optimization")
        sampler = optuna.samplers.TPESampler(
            seed=self.config.seed,
            n_startup_trials=self.config.n_startup_trials,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        best_item: tuple[dict[str, Any], EvaluationResult] | None = None
        remaining = int(self.config.n_trials)
        batch_size = max(1, int(self.config.batch_size))

        while remaining > 0:
            n = min(batch_size, remaining)
            trials = [study.ask() for _ in range(n)]
            candidates = [self._suggest(trial) for trial in trials]
            evaluated = self.evaluator.evaluate(candidates)

            for trial, (x, res) in zip(trials, evaluated):
                trial.set_user_attr("feasible", res.feasible)
                trial.set_user_attr("constraints", dict(res.constraints))
                trial.set_user_attr("objectives", dict(res.objectives))
                trial.set_user_attr("metrics", dict(res.metrics))
                trial.set_user_attr("fea_evaluation_count", int(getattr(self.problem, "fea_evaluation_count", -1)))
                trial.set_user_attr("local_calibration", self.evaluator.local_summary())
                study.tell(trial, float(res.objective))
                if best_item is None or self._rank_key(res) < self._rank_key(best_item[1]):
                    best_item = (dict(x), res)

            remaining -= n

        if best_item is None:
            raise ValueError("Optuna optimization produced no trials.")
        best_x, best_res = best_item
        if self.evaluator.is_synthetic_result(best_res):
            best_x = self.problem.repair(best_x)
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
                    "metrics": dict(tr.user_attrs.get("metrics", {})),
                    "fea_evaluation_count": int(tr.user_attrs.get("fea_evaluation_count", -1)),
                    **self._trial_counts(tr),
                    **dict(tr.user_attrs.get("local_calibration", {})),
                }
            )
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

    def _suggest(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        x = {}
        for spec in self.problem.variables:
            x[spec.name] = trial.suggest_categorical(spec.name, spec.domain)
        return self.problem.repair(x)

    @staticmethod
    def _rank_key(res: EvaluationResult) -> tuple[bool, float]:
        return SurrogateCandidateEvaluator.rank_key(res)

    @staticmethod
    def _trial_counts(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        constraints = dict(trial.user_attrs.get("constraints", {}))
        metrics = dict(trial.user_attrs.get("metrics", {}))
        feasible = bool(trial.user_attrs.get("feasible", False))
        screened = bool(constraints.get("surrogate_screen_reject", 0.0) > 0.0)
        skipped = bool(metrics.get("cost_preselect_skip", False))
        selected = bool(metrics.get("cost_preselect_selected", False))
        return {
            "feasible_count": 1 if feasible else 0,
            "feasible_ratio": 1.0 if feasible else 0.0,
            "screened_count": 1 if screened else 0,
            "screened_ratio": 1.0 if screened else 0.0,
            "cost_preselect_selected_count": 1 if selected else 0,
            "cost_preselect_selected_ratio": 1.0 if selected else 0.0,
            "cost_preselect_skipped_count": 1 if skipped else 0,
            "cost_preselect_skipped_ratio": 1.0 if skipped else 0.0,
        }
