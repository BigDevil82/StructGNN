# Command notebook for surrogate-model and shearwall-optimization runs.
# Usage:
#   1. Open this file.
#   2. Copy the command block you need into a PowerShell terminal.
#   3. Adjust output paths when you want to keep multiple experiment versions.
#
# Long-running commands are intentionally commented out so this file can be
# opened or dot-sourced without accidentally starting training.

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------

# Run from project root:
# cd e:\Common\Desktop\Research\deepLearning\codes\Png2Dxf

# Activate local environment:
# .\.venv\Scripts\Activate.ps1

# Limit BLAS thread oversubscription for heavy batch jobs:
# $env:OMP_NUM_THREADS='1'
# $env:MKL_NUM_THREADS='1'
# $env:OPENBLAS_NUM_THREADS='1'


# ---------------------------------------------------------------------------
# 1. Raw Parametric Dataset Generation
# ---------------------------------------------------------------------------

# Generate FEM samples from layouts.
# python -m scripts.parametric_dataset `
#   --samples-per-layout 5000 `
#   --samples-per-task 50 `
#   --progress-log-interval 500 `
#   --max-workers 160 `
#   --overwrite `
#   --output-dir outputs\result\parametric_dataset_5000x143


# ---------------------------------------------------------------------------
# 2. Surrogate Dataset Preparation
# ---------------------------------------------------------------------------

# Merge raw parametric results into the surrogate sample table.
# python scripts\surrogate\prepare_dataset.py `
#   --input-dir outputs\result\parametric_dataset_5000x143 `
#   --output-dir data\parametric\surrogate_dataset

# Build layout-level scalar features and merge them into samples.
# python scripts\surrogate\build_layout_features.py `
#   --layout-json-dir data\dxf\cad_json_data\fem_raw `
#   --sample-dataset data\parametric\surrogate_dataset\surrogate_samples.parquet `
#   --layout-features-output data\parametric\surrogate_dataset\layout_features.parquet `
#   --merged-output data\parametric\surrogate_dataset\surrogate_samples_with_layout_features.parquet

# Build grouped train/val/test split by layout.
# python scripts\surrogate\make_group_splits.py `
#   --input-path data\parametric\surrogate_dataset\surrogate_samples_with_layout_features.parquet `
#   --output-dir data\parametric\surrogate_dataset\splits


# ---------------------------------------------------------------------------
# 3. Table Baselines for final_pass Classification
# ---------------------------------------------------------------------------

# LightGBM baseline.
# python scripts\surrogate\train_lightgbm.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --output-dir data\parametric\surrogate_dataset\baseline_lightgbm

# CatBoost baseline.
# python scripts\surrogate\train_catboost.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --output-dir data\parametric\surrogate_dataset\baseline_catboost

# MLP embedding k-fold baseline.
# python scripts\surrogate\train_mlp_embedding_kfold.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --fold-path data\parametric\surrogate_dataset\splits\layout_group_folds.parquet `
#   --output-dir data\parametric\surrogate_dataset\baseline_mlp_embedding_kfold


# ---------------------------------------------------------------------------
# 4. GNN final_pass Classification
# ---------------------------------------------------------------------------

# Conservative room-graph GNN used for optimization pre-screening.
# This is the currently recommended screening artifact family.
# python scripts\surrogate\train_gnn_final_pass.py `
#   --graph-repr room `
#   --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --output-dir data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1 `
#   --batch-size 128 `
#   --max-epochs 80 `
#   --early-stop-rounds 10 `
#   --hidden-dim 256 `
#   --gnn-layers 3 `
#   --dropout 0.2 `
#   --lr 0.001 `
#   --weight-decay 0.0001 `
#   --conv-type sage `
#   --monitor-metric screen_reject `
#   --screening-target-recall 0.995 `
#   --log-interval 1

# More aggressive screening version. Higher reject rate, higher false-reject risk.
# python scripts\surrogate\train_gnn_final_pass.py `
#   --graph-repr room `
#   --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --output-dir data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen99_v1 `
#   --batch-size 128 `
#   --max-epochs 80 `
#   --early-stop-rounds 10 `
#   --hidden-dim 256 `
#   --gnn-layers 3 `
#   --dropout 0.2 `
#   --lr 0.001 `
#   --weight-decay 0.0001 `
#   --conv-type sage `
#   --monitor-metric screen_reject `
#   --screening-target-recall 0.99 `
#   --log-interval 1

# Plain F1/PR-AUC-oriented GNN classification experiment.
# python scripts\surrogate\train_gnn_final_pass.py `
#   --graph-repr room `
#   --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --output-dir data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_f1_v1 `
#   --batch-size 128 `
#   --max-epochs 80 `
#   --early-stop-rounds 10 `
#   --hidden-dim 256 `
#   --gnn-layers 3 `
#   --dropout 0.2 `
#   --lr 0.001 `
#   --weight-decay 0.0001 `
#   --conv-type sage `
#   --monitor-metric f1 `
#   --log-interval 1


# ---------------------------------------------------------------------------
# 5. Screening Evaluation and Prediction
# ---------------------------------------------------------------------------

# Predict final_pass on a dataset with a trained GNN artifact.
# python scripts\surrogate\predict_gnn_final_pass.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --artifact-path data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt `
#   --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --output-path data\parametric\surrogate_dataset\screening_predictions.parquet

# Evaluate a prediction file as a feasibility pre-screener.
# python scripts\surrogate\evaluate_screening.py `
#   --predictions-path data\parametric\surrogate_dataset\screening_predictions.parquet `
#   --threshold 0.004


# ---------------------------------------------------------------------------
# 6. Steel Usage Regression
# ---------------------------------------------------------------------------

# LightGBM mean + upper-quantile steel model.
# Used by the current GA surrogate material acceptance path.
# python scripts\surrogate\train_steel_quantile_lightgbm.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --output-dir data\parametric\surrogate_dataset\steel_quantile_lightgbm `
#   --n-estimators 800 `
#   --learning-rate 0.04 `
#   --num-leaves 63 `
#   --subsample 0.85 `
#   --colsample-bytree 0.85 `
#   --upper-alpha 0.95 `
#   --early-stopping-rounds 60

# GNN steel regression. Current best observed steel mean predictor:
# Test MAE ~= 11.5 t, R2 ~= 0.924.
# python scripts\surrogate\train_gnn_steel.py `
#   --graph-repr room `
#   --graph-cache-dir data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --output-dir data\parametric\surrogate_dataset\steel_gnn_room_lr5e4_b512 `
#   --batch-size 512 `
#   --max-epochs 8 `
#   --early-stop-rounds 3 `
#   --lr 0.0005 `
#   --weight-decay 0.0001 `
#   --hidden-dim 128 `
#   --gnn-layers 3 `
#   --conv-type sage `
#   --dropout 0.1 `
#   --log-interval 1

# Quick GNN steel smoke test.
# python scripts\surrogate\train_gnn_steel.py `
#   --output-dir data\parametric\surrogate_dataset\steel_gnn_room_smoke `
#   --batch-size 256 `
#   --max-epochs 3 `
#   --early-stop-rounds 2 `
#   --hidden-dim 128 `
#   --gnn-layers 3 `
#   --dropout 0.15 `
#   --log-interval 1


# ---------------------------------------------------------------------------
# 7. Optimization Runs
# ---------------------------------------------------------------------------

# GA without surrogate acceleration. Baseline for runtime/result comparison.
# python scripts\shearwall_optimize_main.py `
#   --algorithm ga `
#   --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
#   --out outputs\result\optimization\ga_no_surrogate.json `
#   --ga-pop 24 `
#   --ga-gen 20 `
#   --ga-mutation 0.3 `
#   --ga-verbose `
#   --ga-log-every 1

# GA with GNN feasibility pre-screening only.
# python scripts\shearwall_optimize_main.py `
#   --algorithm ga `
#   --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
#   --out outputs\result\optimization\ga_screen995.json `
#   --ga-pop 24 `
#   --ga-gen 20 `
#   --ga-mutation 0.3 `
#   --ga-verbose `
#   --ga-log-every 1 `
#   --ga-surrogate-screen `
#   --ga-surrogate-artifact data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt `
#   --ga-surrogate-graph-cache data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --ga-surrogate-layout-features data\parametric\surrogate_dataset\layout_features.parquet

# GA with screening + conservative material acceptance.
# This currently uses LightGBM steel upper-bound prediction.
# python scripts\shearwall_optimize_main.py `
#   --algorithm ga `
#   --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
#   --out outputs\result\optimization\ga_screen995_material_accept.json `
#   --ga-pop 24 `
#   --ga-gen 20 `
#   --ga-mutation 0.3 `
#   --ga-verbose `
#   --ga-log-every 1 `
#   --ga-surrogate-screen `
#   --ga-surrogate-accept `
#   --ga-surrogate-artifact data\parametric\surrogate_dataset\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt `
#   --ga-surrogate-graph-cache data\parametric\surrogate_dataset\gnn_room_graph_cache `
#   --ga-surrogate-layout-features data\parametric\surrogate_dataset\layout_features.parquet `
#   --ga-steel-artifact-dir data\parametric\surrogate_dataset\steel_quantile_lightgbm `
#   --ga-accept-pass-threshold 0.99 `
#   --ga-accept-max-steel-rel-gap 0.75 `
#   --ga-accept-audit-rate 0.0


# ---------------------------------------------------------------------------
# 8. Result Visualization
# ---------------------------------------------------------------------------

# Visualize an optimization result JSON.
# python scripts\visualize_optimization_result_main.py `
#   --result outputs\result\optimization\ga_screen995_material_accept.json

# Generate surrogate analysis figures.
# python scripts\surrogate\generate_analysis_figures.py `
#   --dataset-path data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet `
#   --predictions-path data\parametric\surrogate_dataset\predictions_lightgbm.parquet `
#   --output-dir outputs\surrogate\analysis_figures



# ---------------------------------------------------------------------------
# 9. GNN steel ranking preselection to GA
# ---------------------------------------------------------------------------

# python scripts\shearwall_optimize_main.py `
#   --algorithm ga `
#   --layout-path data\dxf\cad_json_data\fem_raw\L17_101.json `
#   --N 18 --intensity 6.0 --site-class II --seismic-group 1 `
#   --ga-pop 8 --ga-gen 3 `
#   --optimizer-workers 4 `
#   --ga-steel-ranking `
#   --ga-steel-ranking-eval-ratio 0.5 `
#   --ga-steel-ranking-min-eval 4 `
#   --ga-steel-ranking-random-ratio 0.125 `
#   --ga-steel-ranking-artifact data\parametric\ckpt\steel_gnn_room_lr5e4_b512\gnn_steel.pt `
#   --ga-steel-ranking-graph-cache data\parametric\cache\gnn_room_graph_cache