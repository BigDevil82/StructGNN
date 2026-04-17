import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.shearwall_modeling.parametric import DatasetGenerationConfig

from ..algorithms import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmOptimizer,
    OptunaBayesConfig,
    OptunaBayesOptimizer,
    ParticleSwarmConfig,
    ParticleSwarmOptimizer,
    RandomSearchConfig,
    RandomSearchOptimizer,
)
from ..core.contracts import OptimizationResult
from ..problems import (
    ShearWallConstraintConfig,
    ShearWallLimitConfig,
    ShearWallObjectiveConfig,
    ShearWallOptimizationProblem,
)


def run_shearwall_optimization(
    layout_path: str,
    out_path: str,
    *,
    algorithm: str = "ga",
    fixed_params: dict[str, Any] | None = None,
    decision_space: dict[str, list[Any]] | None = None,
    analysis_cfg: DatasetGenerationConfig | None = None,
    objective_cfg: ShearWallObjectiveConfig | None = None,
    constraint_cfg: ShearWallConstraintConfig | None = None,
    limit_cfg: ShearWallLimitConfig | None = None,
    ga_cfg: GeneticAlgorithmConfig | None = None,
    optuna_cfg: OptunaBayesConfig | None = None,
    pso_cfg: ParticleSwarmConfig | None = None,
    random_cfg: RandomSearchConfig | None = None,
) -> OptimizationResult:
    cfg = analysis_cfg or DatasetGenerationConfig(samples_per_layout=1, samples_per_task=1, max_workers=0)
    problem = ShearWallOptimizationProblem(
        layout_path=layout_path,
        analysis_cfg=cfg,
        fixed_params=fixed_params,
        decision_space=decision_space,
        objective_cfg=objective_cfg,
        constraint_cfg=constraint_cfg,
        limit_cfg=limit_cfg,
    )

    if algorithm == "ga":
        optimizer = GeneticAlgorithmOptimizer(problem, ga_cfg)
    elif algorithm == "optuna":
        optimizer = OptunaBayesOptimizer(problem, optuna_cfg)
    elif algorithm == "pso":
        optimizer = ParticleSwarmOptimizer(problem, pso_cfg)
    elif algorithm == "random":
        optimizer = RandomSearchOptimizer(problem, random_cfg)
    else:
        raise ValueError(f"Unsupported algorithm={algorithm}")

    result = optimizer.optimize()

    payload = {
        "algorithm": algorithm,
        "layout_path": layout_path,
        "analysis_cfg": asdict(cfg),
        "best_solution": result.best_solution,
        "best_objective": result.best_objective,
        "best_objectives": result.best_objectives,
        "best_feasible": result.best_feasible,
        "best_constraints": result.best_constraints,
        "history": result.history,
    }
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
