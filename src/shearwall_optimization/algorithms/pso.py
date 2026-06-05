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
class ParticleSwarmConfig:
    swarm_size: int = 24
    iterations: int = 20
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    velocity_clamp: float = 0.25
    mutation_rate: float = 0.0
    max_workers: int | None = None
    seed: int = 42
    surrogate_screening: SurrogateScreeningConfig | None = None
    cost_preselection: SurrogateCostPreselectionConfig | None = None
    local_calibration: OnlineLocalCalibrationConfig | None = None


class ParticleSwarmOptimizer(Optimizer):
    def __init__(self, problem, config: ParticleSwarmConfig | None = None):
        super().__init__(problem)
        self.config = config or ParticleSwarmConfig()
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
        print("[PSO] Starting optimization")
        dim = len(self.problem.variables)
        particles = [[self.rng.random() for _ in range(dim)] for _ in range(self.config.swarm_size)]
        velocities = [[0.0 for _ in range(dim)] for _ in range(self.config.swarm_size)]

        pbest_pos = [list(p) for p in particles]
        pbest_eval = self._evaluate_vectors(particles)

        gbest_idx = min(range(self.config.swarm_size), key=lambda i: self._rank_key(pbest_eval[i]))
        gbest_pos = list(pbest_pos[gbest_idx])
        gbest_eval = pbest_eval[gbest_idx]

        history: list[dict[str, Any]] = []
        for it in range(1, self.config.iterations + 1):
            mutated_count = 0
            for i in range(self.config.swarm_size):
                for d in range(dim):
                    r1 = self.rng.random()
                    r2 = self.rng.random()
                    v = (
                        self.config.inertia * velocities[i][d]
                        + self.config.cognitive * r1 * (pbest_pos[i][d] - particles[i][d])
                        + self.config.social * r2 * (gbest_pos[d] - particles[i][d])
                    )
                    v = max(-self.config.velocity_clamp, min(self.config.velocity_clamp, v))
                    velocities[i][d] = v
                    particles[i][d] = max(0.0, min(1.0, particles[i][d] + v))
                    if self.rng.random() < self.config.mutation_rate:
                        particles[i][d] = self.rng.random()
                        velocities[i][d] = 0.0
                        mutated_count += 1

            iter_eval = self._evaluate_vectors(particles)
            for i, cur_eval in enumerate(iter_eval):
                if self._rank_key(cur_eval) < self._rank_key(pbest_eval[i]):
                    pbest_pos[i] = list(particles[i])
                    pbest_eval[i] = cur_eval

                if self._rank_key(cur_eval) < self._rank_key(gbest_eval):
                    gbest_pos = list(particles[i])
                    gbest_eval = cur_eval

            counts = self.evaluator.population_counts(iter_eval)
            history.append(
                {
                    "iteration": it,
                    "best_objective": gbest_eval.objective,
                    "best_feasible": gbest_eval.feasible,
                    **counts,
                    "fea_evaluation_count": int(getattr(self.problem, "fea_evaluation_count", -1)),
                    "pso_mutated_coordinate_count": mutated_count,
                    "pso_mutation_rate": self.config.mutation_rate,
                    **self.evaluator.local_summary(),
                    "population": [self.evaluator.population_item(r) for r in iter_eval],
                    "best_constraints": dict(gbest_eval.constraints),
                }
            )

        best_solution = self.problem.decode(gbest_pos)
        if self.evaluator.is_synthetic_result(gbest_eval):
            best_solution = self.problem.repair(best_solution)
            gbest_eval = self.problem.evaluate(best_solution)
        if history:
            history[-1]["final_fea_evaluation_count"] = int(getattr(self.problem, "fea_evaluation_count", -1))
        return OptimizationResult(
            best_solution=best_solution,
            best_objective=gbest_eval.objective,
            best_objectives=gbest_eval.objectives,
            best_feasible=gbest_eval.feasible,
            best_constraints=gbest_eval.constraints,
            history=history,
        )

    def _evaluate_vectors(self, vectors: list[list[float]]):
        decoded = [self.problem.decode(v) for v in vectors]
        return [res for _, res in self.evaluator.evaluate(decoded)]

    @staticmethod
    def _rank_key(res: EvaluationResult) -> tuple[bool, float]:
        return SurrogateCandidateEvaluator.rank_key(res)
