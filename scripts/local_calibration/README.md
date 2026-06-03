# Local Steel Residual Calibration

This folder contains experimental scripts for layout-local calibration of a global steel-usage surrogate.

The current validation question is:

> Given a fixed layout and a small number of fresh FEA results, can a simple local model learn the residual between true steel usage and the offline global surrogate prediction?

## Script

`steel_residual_calibration.py` merges:

- global steel predictions, default `data\parametric\ckpt\steel_gnn_room_lr5e4_b512\predictions_test.parquet`
- design variables from the surrogate parquet, default `data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet`

For each layout, it randomly selects `calib_n` samples as local calibration data and evaluates on the remaining samples from the same layout.

Supported local residual models:

- `ridge`
- `rf`
- `lgbm`
- `gpr`
- `knn`
- `mlp`

Inputs can be controlled with `--input-mode`:

- `pred`: only global steel prediction
- `param`: only design variables
- `pred_param`: global prediction plus design variables

Residual target can be controlled with `--target-space`:

- `kg`: learn `y_true - y_pred`
- `log`: learn `log1p(y_true) - log1p(y_pred)`

## Quick Run

```powershell
.\.venv\Scripts\python.exe scripts\local_calibration\steel_residual_calibration.py `
  --max-layouts 6 `
  --calib-sizes 25,50,100,200,500 `
  --repeats 3 `
  --models ridge,rf,lgbm,gpr,knn `
  --output-dir outputs\result\local_calibration\steel_residual_quick
```

## Full Run

```powershell
.\.venv\Scripts\python.exe scripts\local_calibration\steel_residual_calibration.py `
  --layouts all `
  --calib-sizes 25,50,100,200,500,1000 `
  --repeats 5 `
  --models ridge,rf,lgbm,gpr,knn,mlp `
  --output-dir outputs\result\local_calibration\steel_residual_full
```

`gpr` is cubic in the number of calibration samples, so the script caps its training subset with `--max-gpr-train`, default `300`.

## Outputs

- `results.csv`: every layout/repeat/calibration-size/model result.
- `summary.csv`: mean metrics grouped by calibration size and model.
- `best_by_layout.csv`: best calibrated model for each layout and calibration size.
- `report.md`: compact markdown summary of the best mean result by calibration size.

The key columns are:

- `mae`, `rmse`, `r2`, `mape`, `bias`
- `mae_improve_pct`, `rmse_improve_pct`, `mape_improve_pct`

A positive improvement means the local residual model improves over the original global surrogate on the same holdout samples.

## Feasibility Probability Calibration

`feasibility_probability_calibration.py` tests whether a layout-local calibration set can improve the feasibility classifier probability.

Supported methods:

- `platt`: logistic calibration on the global probability logit.
- `isotonic`: monotonic probability calibration.
- `logistic`: local logistic classifier using probability plus design variables.
- `residual_ridge`, `residual_rf`, `residual_lgbm`: learn `y_true - p_global` and correct the probability.
- `vote`: average global probability with several local classifiers.

Quick run:

```powershell
.\.venv\Scripts\python.exe scripts\local_calibration\feasibility_probability_calibration.py `
  --max-layouts 6 `
  --calib-sizes 25,50,100,200,500 `
  --repeats 3 `
  --screening-threshold-scale 0.1 `
  --output-dir outputs\result\local_calibration\feasibility_probability_quick_scale01
```

Full run:

```powershell
.\.venv\Scripts\python.exe scripts\local_calibration\feasibility_probability_calibration.py `
  --layouts all `
  --calib-sizes 25,50,100,200,500,1000 `
  --repeats 5 `
  --screening-threshold-scale 0.1 `
  --output-dir outputs\result\local_calibration\feasibility_probability_full_scale01
```

The main probability-quality metrics are `brier` and `ece`.
For optimization screening, inspect `screen_reject_rate` together with `screen_recall`.
`--screening-threshold-scale` makes the local reject threshold more conservative; values below `1.0` reduce false rejection risk at the cost of rejecting fewer candidates.
