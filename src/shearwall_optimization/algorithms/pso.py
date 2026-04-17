import random
from dataclasses import dataclass
from typing import Any

from ..core.contracts import OptimizationResult
from ..core.optimizer import Optimizer


@dataclass(frozen=True)
class ParticleSwarmConfig:
    swarm_size: int = 24
    iterations: int = 20
    inertia: float = 0.72
    cognitive: float = 1.49
    social: float = 1.49
    velocity_clamp: float = 0.25
    max_workers: int | None = None
    seed: int = 42


class ParticleSwarmOptimizer(Optimizer):
    def __init__(self, problem, config: ParticleSwarmConfig | None = None):
        super().__init__(problem)
        self.config = config or ParticleSwarmConfig()
        self.rng = random.Random(self.config.seed)

    def optimize(self) -> OptimizationResult:
        dim = len(self.problem.variables)
        particles = [[self.rng.random() for _ in range(dim)] for _ in range(self.config.swarm_size)]
        velocities = [[0.0 for _ in range(dim)] for _ in range(self.config.swarm_size)]

        pbest_pos = [list(p) for p in particles]
        pbest_eval = self._evaluate_vectors(particles)

        gbest_idx = min(range(self.config.swarm_size), key=lambda i: pbest_eval[i].objective)
        gbest_pos = list(pbest_pos[gbest_idx])
        gbest_eval = pbest_eval[gbest_idx]

        history: list[dict[str, Any]] = []
        for it in range(1, self.config.iterations + 1):
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

            iter_eval = self._evaluate_vectors(particles)
            for i, cur_eval in enumerate(iter_eval):
                if cur_eval.objective < pbest_eval[i].objective:
                    pbest_pos[i] = list(particles[i])
                    pbest_eval[i] = cur_eval

                if cur_eval.objective < gbest_eval.objective:
                    gbest_pos = list(particles[i])
                    gbest_eval = cur_eval

            history.append(
                {
                    "iteration": it,
                    "best_objective": gbest_eval.objective,
                    "best_feasible": gbest_eval.feasible,
                    "best_constraints": dict(gbest_eval.constraints),
                }
            )

        best_solution = self.problem.decode(gbest_pos)
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
        if hasattr(self.problem, "evaluate_many"):
            return self.problem.evaluate_many(decoded, max_workers=self.config.max_workers)
        return [self.problem.evaluate(x) for x in decoded]
