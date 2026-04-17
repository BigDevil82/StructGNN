# Shearwall Optimization Framework

This package isolates optimization logic from structural modeling and analysis.

## Structure

- `core/`: abstract contracts (`OptimizationProblem`, `Optimizer`) and result dataclasses.
- `problems/`: problem adapters that bind design variables to the shared modeling-analysis pipeline.
- `algorithms/`: traditional optimization algorithms (currently GA, PSO, Optuna Bayesian optimization, and Random Search).
- `runners/`: high-level orchestration and result serialization.

## Reuse with Shared Modeling-Analysis Pipeline

`problems.shearwall_problem.ShearWallOptimizationProblem` reuses:
- parametric model configuration builder
- shared OpenSees analysis execution
- shared evaluation and reinforcement design checks

This ensures optimization and dataset generation use the same physics/check logic.

## Extension Plan

- Add new algorithm in `algorithms/` by implementing `Optimizer` contract.
- Add new problem variants in `problems/` (multi-layout, multi-objective, robustness optimization).
- Add RL in a separate package (recommended: `src/shearwall_optimization_rl/`) while reusing `problems/` and shared analysis API.

## Parallel Evaluation

- GA and PSO support process-parallel candidate evaluation through `evaluate_many`.
- CLI usage: set `--optimizer-workers N` in `scripts.shearwall_optimize_main`.
- Keep `--optimizer-workers 0` for sequential mode.

## Multi-objective Interface

- `EvaluationResult.objectives` now includes:
	- `material_cost`
	- `margin_penalty`
- Scalar objective used by GA/PSO is:
	- `material_cost + margin_weight * margin_penalty + infeasible_penalty * violation_sum`
- CLI can set `--objective-margin-weight` and limit thresholds to tune this scalarization.
