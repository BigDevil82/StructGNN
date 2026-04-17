# Shearwall Optimization Framework

This package isolates optimization logic from structural modeling and analysis.

## Structure

- `core/`: abstract contracts (`OptimizationProblem`, `Optimizer`) and result dataclasses.
- `problems/`: problem adapters that bind design variables to the shared modeling-analysis pipeline.
- `algorithms/`: traditional optimization algorithms (currently GA and Random Search).
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
