# Paper2 Experiments

This folder contains paper-oriented experiment scripts. The scripts are not part of the core surrogate or optimization package. They read existing model artifacts and produce compact tables and figures for the manuscript.

## Section 3.2 Surrogate Model Performance

Run:

```powershell
.\.venv\Scripts\python.exe docs\paper2\experiment\surrogate_model_performance.py
```

Default inputs:

- feasibility surrogate: `data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1`
- steel param-only baseline: `data\parametric\ckpt\steel_ablation_param_only`
- steel param + layout-stat baseline: `data\parametric\ckpt\steel_ablation_param_graph_feat`
- steel LayoutParamGNN: `data\parametric\ckpt\steel_gnn_room_lr5e4_b512`
- dataset: `data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet`

Outputs:

- `outputs\result\paper2\surrogate_model_performance\tables\feasibility_surrogate_metrics.csv`
- `outputs\result\paper2\surrogate_model_performance\tables\feasibility_screening_curve.csv`
- `outputs\result\paper2\surrogate_model_performance\tables\feasibility_error_by_layout.csv`
- `outputs\result\paper2\surrogate_model_performance\tables\steel_surrogate_metrics.csv`
- `outputs\result\paper2\surrogate_model_performance\tables\steel_error_by_true_quantile.csv`
- `outputs\result\paper2\surrogate_model_performance\tables\steel_error_by_layout.csv`
- `outputs\result\paper2\surrogate_model_performance\plots\main_3_2_surrogate_model_performance.png`
- `outputs\result\paper2\surrogate_model_performance\report.md`

The generated figure combines:

- feasibility precision-recall curve for overall test performance;
- layout-level feasibility probability bias, showing where the global classifier is less reliable;
- steel predicted-vs-true scatter for overall regression performance;
- steel error by true-usage quantile, showing systematic bias at high steel usage.

This figure is intended to support the Section 3.2 narrative: the two global surrogates are useful but imperfect. Their layout-level and target-range-dependent errors motivate the online local calibration experiments in Section 3.3.

To compare another model artifact, pass model specs as `label=artifact_dir`:

```powershell
.\.venv\Scripts\python.exe docs\paper2\experiment\surrogate_model_performance.py `
  --feasibility-models "LayoutParamGNN=data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1" `
  --steel-models `
    "Param only=data\parametric\ckpt\steel_ablation_param_only" `
    "Param + layout stats=data\parametric\ckpt\steel_ablation_param_graph_feat" `
    "LayoutParamGNN=data\parametric\ckpt\steel_gnn_room_lr5e4_b512"
```

## Section 3.3 Online Local Calibration

Generate the standard local-calibration data:

```powershell
.\.venv\Scripts\python.exe docs\paper2\experiment\run_local_calibration_paper_experiment.py
```

Default outputs:

- `outputs\result\paper2\local_calibration\feasibility\results.csv`
- `outputs\result\paper2\local_calibration\feasibility\summary.csv`
- `outputs\result\paper2\local_calibration\steel\results.csv`
- `outputs\result\paper2\local_calibration\steel\summary.csv`
- `outputs\result\paper2\local_calibration\commands.json`

The script reuses the existing calibration implementations in `scripts\local_calibration`:

- feasibility probability calibration: Platt, isotonic, logistic, and LightGBM residual calibration;
- steel residual calibration: Ridge, Gaussian process, LightGBM, and KNN.

For a quick smoke run:

```powershell
.\.venv\Scripts\python.exe docs\paper2\experiment\run_local_calibration_paper_experiment.py `
  --out-root outputs\result\paper2\local_calibration_smoke `
  --max-layouts 3 `
  --calib-sizes 25,50 `
  --repeats 1
```

Plot the paper figures:

```powershell
.\.venv\Scripts\python.exe docs\paper2\plot\plot_local_calibration_paper_figures.py `
  --result-root outputs\result\paper2\local_calibration `
  --out-dir outputs\result\paper2\local_calibration\plots `
  --calib-n 100
```

Generated figures:

- `main_3_3_local_calibration_sample_efficiency.png`
- `main_3_3_local_calibration_layout_improvement.png`
