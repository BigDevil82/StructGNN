# Shearwall Optimization Framework

This package isolates optimization logic from structural modeling and analysis.

## Structure

- `core/`: abstract contracts (`OptimizationProblem`, `Optimizer`) and result dataclasses.
- `problems/`: problem adapters that bind design variables to the shared modeling-analysis pipeline.
- `algorithms/`: traditional optimization algorithms (currently GA, PSO, Optuna Bayesian optimization, NSGA-II, and Random Search).
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

## GA Surrogate Pre-screening

GA can optionally use the trained GNN screening model before FEM evaluation:

```powershell
python scripts\shearwall_optimize_main.py `
  --algorithm ga `
  --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
  --ga-surrogate-screen
```

When enabled, each generation is screened in batch. Candidates with `pred_screen_reject=True`
are assigned an infeasible `EvaluationResult` with constraint `surrogate_screen_reject=1.0`
and are not sent to FEM. The optimization history records `screened_count` and
`screened_ratio` per generation, allowing direct comparison against runs without
`--ga-surrogate-screen`.

The default artifact is the conservative screening model:

```text
data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt
```

Use `--ga-surrogate-threshold` to override the artifact's screening threshold for
more conservative or more aggressive filtering.

## GA Surrogate Material Acceptance

GA can also skip FEM for high-confidence feasible candidates by combining:

- GNN pass probability from `GNNFeasibilityScreener`
- LightGBM steel usage mean/upper-quantile prediction
- fast concrete quantity estimate from layout scalar features and section sizes

Train the steel surrogate first:

```powershell
python scripts\surrogate\train_steel_quantile_lightgbm.py
```

Then enable the second-stage accelerator:

```powershell
python scripts\shearwall_optimize_main.py `
  --algorithm ga `
  --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
  --ga-surrogate-screen `
  --ga-surrogate-accept `
  --ga-accept-pass-threshold 0.99 `
  --ga-accept-max-steel-rel-gap 0.75
```

Accepted candidates are marked with `metrics.surrogate_material_accept=True`.
Their objective uses the conservative steel upper bound rather than the mean
prediction. GA verifies the final best solution with the real FEM evaluator if
that best solution came from surrogate material acceptance.

## Multi-objective Interface

- `EvaluationResult.objectives` now includes:
	- `material_cost`
	- `margin_penalty`
- Scalar objective used by GA/PSO is:
	- `material_cost + margin_weight * margin_penalty + infeasible_penalty * violation_sum`
- CLI can set `--objective-margin-weight` and limit thresholds to tune this scalarization.
