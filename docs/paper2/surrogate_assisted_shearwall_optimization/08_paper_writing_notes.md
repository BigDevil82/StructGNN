# Paper writing notes

## Suggested contribution statement

The paper can claim the following contributions:

1. A parametric OpenSees-based workflow for shear wall section design, integrating structural response analysis, code-oriented feasibility evaluation, and reinforcement quantity estimation.
2. A layout-aware surrogate modeling framework for feasibility screening and steel-usage prediction based on sampled section/material/design-condition data.
3. An online local calibration strategy that adapts global surrogate predictions to the current layout during optimization.
4. A surrogate-assisted optimization framework that reduces unnecessary FEA calls while preserving final high-fidelity verification.

Avoid claiming that the GNN alone is the main innovation unless final experiments clearly show a strong standalone advantage over tabular baselines.

## Recommended title direction

Possible title style:

```text
Surrogate-assisted section optimization of shear wall structures with online local calibration
```

This title emphasizes the full framework rather than only the neural model.

## How to position the surrogate models

The feasibility model should be described as a conservative screener:

- not a binary replacement for design checking;
- optimized for high recall of feasible designs;
- used to reject only clearly infeasible candidates.

The steel model should be described as a cost-ranking model:

- not used to certify the final material cost;
- used to allocate FEA budget among retained candidates;
- combined with concrete-cost estimation and feasibility-risk penalty.

## How to discuss difficult layouts

Some layouts may be infeasible under the available section search space. This should not be framed as method failure if full FEA optimization also fails to find feasible designs.

A fair interpretation is:

- the proposed method should be compared against full FEA under identical search space and conditions;
- if both methods fail, the layout/design condition may be outside the feasible region;
- if the surrogate method fails while full FEA succeeds, this is a meaningful miss and should be analyzed.

## What not to overemphasize

Avoid overloading the paper with every explored variant:

- many GNN pooling variants;
- all intermediate failed classifiers;
- pairwise ranking experiments unless used in final method;
- every PSO parameter trial.

These can be briefly mentioned as development checks, but the paper should focus on the final framework and key ablations.

## Core narrative

The clean narrative is:

1. Full FEA-based optimization is accurate but computationally expensive.
2. Offline surrogate models can identify clearly poor candidates and estimate material cost.
3. Pure surrogate decisions are risky because prediction error is layout-dependent.
4. Online local calibration reduces this risk using FEA samples already generated during optimization.
5. The resulting framework spends FEA calls on more promising candidates and verifies the final design with the original high-fidelity workflow.

## Recommended terminology

Use consistent terms:

- high-fidelity evaluator: OpenSees analysis + design workflow;
- global surrogate: model trained on the full offline dataset;
- local calibrator: online residual/probability correction model for one layout;
- screening: rejecting clearly infeasible candidates;
- preselection: selecting a subset of retained candidates for FEA based on predicted cost;
- real FEA calls: unique high-fidelity evaluations after cache deduplication.

