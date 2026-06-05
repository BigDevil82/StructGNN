# Experiment setup

## Purpose

The experiments should demonstrate that surrogate-assisted optimization reduces FEA cost while preserving optimization quality.

The core hypothesis is:

```text
Using conservative feasibility screening and calibrated cost preselection can reduce real FEA calls without materially reducing feasible-solution discovery or final material-cost quality.
```

## Dataset experiments

The surrogate-model experiments should be designed to answer:

1. Can the model identify clearly infeasible candidates with high feasible recall?
2. Can the model predict steel usage accurately enough for candidate ranking?
3. Does layout information improve performance beyond design parameters alone?

Recommended comparisons:

- parameter-only MLP;
- parameter + layout statistics model;
- room-graph GNN;
- deployed conservative feasibility screener;
- deployed steel-usage predictor.

Metrics for feasibility screening:

- feasible recall;
- false rejection rate;
- reject rate;
- reject-infeasible precision;
- PR-AUC and ROC-AUC as secondary metrics.

Metrics for steel prediction:

- MAE;
- RMSE;
- MAPE;
- R2;
- error by steel-mass quantile;
- layout-level mean absolute error.

## Optimization experiments

The primary optimization comparison should use the same layouts, seeds, and design conditions for all methods.

Recommended main methods:

| Method | Description |
| --- | --- |
| `full` | full FEA evaluation for all candidates |
| `gnn_screen_cost` | proposed surrogate-assisted method |
| `gnn_cost` | ablation without feasibility screening |

The `gnn_cost` ablation is important because it shows that steel-cost prediction alone is insufficient when low-cost candidates may be infeasible.

## Layout selection

Two experimental scopes are useful:

1. **Primary test layouts**: selected unseen test layouts from `L17`, `L27`, and `L1L28`, each with multiple random seeds.
2. **All-layout feasibility scan**: all 143 layouts with one seed, used to understand which layouts are feasible under the current section search space.

The all-layout scan should be treated as dataset characterization, not as the main surrogate comparison.

## Fixed seismic conditions

For primary experiments:

- `L17`: intensity 7.0;
- `L27`: intensity 7.5;
- `L1L28`: intensity 8.0.

For each layout-seed pair, site class and seismic group are sampled once and reused across compared methods. This avoids confounding method performance with different design conditions.

## GA configuration

The main GA configuration used in recent experiments is:

```text
population = 48
generations = 15
elite_size = 2
mutation_rate = 0.3
```

The implementation caches repeated designs, so the number of real FEA calls can be lower than the nominal number of generated candidates.

## Surrogate-assisted configuration

The key surrogate optimization parameters are:

```text
eval_ratio = 0.5
min_eval_count = 8
feasibility_penalty_cost = 1.0e6
feasibility_hinge_target = 0.5
local_calibration_min_samples = 25
local_screening_threshold_scale = 0.1
```

These values should be reported because they define the tradeoff between FEA reduction and risk of missing feasible designs.

## Additional optimizer comparison

If included, PSO and random search should be framed as secondary validation:

- random search tests whether screening alone can allocate expensive evaluations more effectively;
- PSO tests transferability to a population-based optimizer with different update dynamics.

For PSO, mention that discrete decoding can cause premature convergence and repeated candidates. A small mutation rate can improve exploration, but GA remains the main method for discrete section optimization.

## Result reporting plan

The results section can be organized around the following tables and figures:

1. surrogate screening performance table;
2. steel prediction performance table;
3. optimization method summary table;
4. paired comparison against full FEA optimization;
5. FEA-call reduction distribution;
6. final material-cost ratio distribution among both-feasible cases;
7. layout-level feasibility and difficulty analysis.

This document intentionally does not include numerical results. Those should be added after selecting the final experimental run.

