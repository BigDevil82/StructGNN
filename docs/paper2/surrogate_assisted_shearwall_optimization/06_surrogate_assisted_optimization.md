# Surrogate-assisted optimization

## Baseline optimization

The baseline optimizer evaluates every generated candidate with the full OpenSees workflow. For each candidate:

1. repair the design variables;
2. build the structural model;
3. run FEA and seismic evaluation;
4. run reinforcement design;
5. compute material cost and constraints;
6. rank feasible candidates ahead of infeasible candidates.

This baseline is accurate but expensive.

## Unified candidate evaluator

To avoid rewriting surrogate logic for every optimizer, the method introduces a unified candidate evaluator. The evaluator receives a batch of candidate designs and returns evaluation results in the same format as a real FEA call.

The evaluator can operate in three modes:

1. **Full mode**: all candidates are evaluated by FEA.
2. **Cost preselection mode**: all candidates receive surrogate cost predictions, but only a subset is evaluated by FEA.
3. **Screen + cost mode**: clearly infeasible candidates are rejected first; the remaining candidates are ranked by calibrated predicted cost and feasibility risk.

This abstraction allows GA, PSO, Optuna/TPE, and random search to reuse the same screening and preselection logic.

## Feasibility screening

For each candidate, the feasibility screener returns:

```text
probability, threshold, reject
```

If `reject = true`, the candidate is assigned a synthetic infeasible result:

```text
surrogate_screen_reject = 1
objective = rejected_objective
```

It is not sent to FEA. This reduces computation for candidates that are very likely infeasible.

## Cost-based preselection

For candidates not rejected by screening, the steel surrogate predicts steel usage. Concrete usage is estimated from geometry and sections. These are combined into a predicted material cost.

Candidates are ranked by:

```text
score = predicted_material_cost + feasibility_penalty * infeasible_risk
```

where:

```text
infeasible_risk = max(0, hinge_target - feasible_probability)
```

Only the top candidates are sent to FEA:

```text
eval_count = max(min_eval_count, round(eval_ratio * candidate_count))
```

The remaining candidates receive a synthetic skipped result:

```text
cost_preselect_skip = true
objective = skipped_objective
```

## Final verification

If the best candidate at the end of optimization is synthetic, it is repaired and evaluated by the real FEA workflow before being returned. Therefore, the reported final best solution is always backed by real analysis.

This rule is important for paper credibility: surrogate models reduce intermediate evaluation cost, but they do not produce unverified final designs.

## Genetic algorithm workflow

The GA is the main optimization algorithm for the paper because the design variables are discrete.

The GA uses:

- initial random population;
- feasible-first sorting;
- elite preservation;
- tournament selection;
- uniform crossover;
- random mutation;
- batch evaluation through the unified evaluator.

Surrogate-assisted GA differs from full GA only in the population evaluation stage. Selection, crossover, mutation, and final ranking remain unchanged.

## Other optimizers

Other optimizers are included to test whether the screening idea is algorithm-independent:

- **PSO**: continuous particle positions are decoded into discrete section choices; mutation can be added to reduce premature convergence in the discrete space.
- **Random search**: random batches are filtered or ranked by the same evaluator.
- **Optuna/TPE**: candidates are suggested by a Bayesian/TPE sampler and then evaluated through the same surrogate-aware evaluator.

For the paper, these methods should be used as supporting evidence that the framework is not tied to GA. GA should remain the primary demonstration because it is more naturally suited to discrete section optimization.

## Method naming in experiments

Use clear method names:

| Method | Meaning |
| --- | --- |
| `full` | every candidate evaluated by FEA |
| `gnn_cost` | cost preselection using steel surrogate, without feasibility screening |
| `gnn_screen_cost` | feasibility screening + cost preselection + optional online calibration |

The main proposed method is `gnn_screen_cost`.

## Evaluation focus

The optimization method should be judged by system-level metrics:

- final feasibility rate;
- final material cost among feasible runs;
- objective ratio relative to full FEA optimization;
- number of real FEA calls;
- FEA reduction rate;
- first feasible FEA call count;
- robustness across layouts and random seeds.

Do not evaluate the method only by surrogate model F1 or regression MAE. Those metrics are auxiliary.

