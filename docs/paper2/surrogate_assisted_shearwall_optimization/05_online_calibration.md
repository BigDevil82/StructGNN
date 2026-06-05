# Online local calibration

## Motivation

The offline surrogate models are trained across all layouts and design conditions. During optimization, however, the search is performed for one fixed layout and one fixed seismic condition at a time.

This creates an opportunity: every FEA evaluation produced during optimization becomes a local labeled sample for that specific layout. These local samples can be used to calibrate the global surrogate predictions online.

The purpose of online calibration is not to retrain the whole GNN. Instead, it learns a lightweight local correction model:

```text
local prediction = global prediction + local residual correction
```

## Calibration data

For every candidate that is actually evaluated by FEA, the optimizer records:

- design variables;
- true feasibility result;
- global feasibility probability, if available;
- global predicted steel mass, if available;
- true steel mass from FEA and reinforcement design.

Repeated designs are stored by a decision-key cache, so each unique candidate contributes at most one local record.

## Feasibility calibration

The feasibility model uses Platt-style calibration. The input is the global model probability `p`. It is transformed by a logit function:

```text
z = log(p / (1 - p))
```

A logistic regression model is then fit:

```text
P(local feasible | p) = sigmoid(a z + b)
```

This calibration starts only after enough local samples are available and both feasible and infeasible classes have been observed.

## Conservative screening threshold

The calibrated threshold is selected from local positive samples. Given a target recall:

```text
screening_target_recall = 0.995
```

the threshold is chosen near the lower tail of calibrated probabilities among locally feasible samples. The implementation further scales the threshold:

```text
threshold = quantile_positive_probability * screening_threshold_scale
```

The default scale is conservative:

```text
screening_threshold_scale = 0.1
```

This makes local calibration less likely to reject feasible candidates too early.

## Steel residual calibration

The steel calibration model predicts residual error:

```text
r = true_steel_kg - global_predicted_steel_kg
```

The calibrated prediction is:

```text
calibrated_steel_kg = max(global_predicted_steel_kg + r_hat, 0)
```

The local feature vector includes:

- design variables;
- global predicted steel mass;
- log-transformed global predicted steel mass.

The model type depends on the number of local samples:

| Local sample count | Model |
| --- | --- |
| small sample count | Ridge regression |
| medium sample count | Gaussian process regression |
| large sample count | LightGBM regression |

This staged design avoids using a high-capacity local model before enough FEA samples have accumulated.

## Online update loop

Within each optimization generation or batch:

1. the global surrogate predicts feasibility and steel mass;
2. local calibration adjusts probability and steel prediction if enough local samples exist;
3. selected candidates are evaluated by FEA;
4. FEA results are added to the local calibration set;
5. calibration models are refit lazily when new records are available.

## Paper framing

The online calibration module should be presented as an adaptive layer on top of a global surrogate model:

- the global model provides broad generalization across layouts;
- the local calibration model adapts to the current layout and design condition;
- only real FEA samples generated during optimization are used;
- no additional offline dataset is required for calibration.

This supports the paper's key claim that the method is not merely a static surrogate, but a surrogate-assisted optimization framework that improves as the optimizer samples the local design space.

