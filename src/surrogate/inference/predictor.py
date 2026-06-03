from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import Pool
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.surrogate.features.consts import LAYOUT_FEATURES, PARAM_FEATURES
from src.surrogate.training.lightgbm_baseline import REG_TASKS
from src.surrogate.training.mlp_embedding_kfold import EmbeddingMLP, predict_prob, transform_with_preprocess

CAT_COLS = ["conc_bot", "site_class", "intensity", "seismic_group"]


def predict_with_lightgbm(
    df: pd.DataFrame,
    artifact_dir: str | Path,
    threshold: float | None = None,
    report_metrics: bool = True,
) -> pd.DataFrame:
    feature_cols = PARAM_FEATURES + LAYOUT_FEATURES
    _check_required_columns(df, feature_cols)

    artifact = Path(artifact_dir)
    feature_path = artifact / "feature_columns.json"
    if not feature_path.exists():
        raise ValueError(f"Missing feature_columns.json in {artifact}.")

    trained_cols = json.loads(feature_path.read_text(encoding="utf-8"))
    x = _prepare_lightgbm_features(df, feature_cols, trained_cols)

    clf = joblib.load(artifact / "clf_final_pass_lightgbm.joblib")
    reg_models = {target: joblib.load(artifact / f"reg_{target}_lightgbm.joblib") for target in REG_TASKS}

    threshold = _load_threshold(artifact / "metrics.json") if threshold is None else threshold

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = clf.predict_proba(x)[:, 1]
    pred["pred_final_pass"] = pred["pred_final_pass_prob"] >= threshold
    for target, model in reg_models.items():
        pred[f"pred_{target}"] = model.predict(x)

    if report_metrics:
        metrics = _collect_prediction_metrics(df, pred)
        if metrics:
            print("[surrogate][lightgbm] prediction metrics")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return pred


def predict_with_mlp_embedding_kfold(
    df: pd.DataFrame,
    artifact_dir: str | Path,
    threshold: float | None = None,
    report_metrics: bool = True,
    batch_size: int = 4096,
) -> pd.DataFrame:
    artifact = Path(artifact_dir)
    fold_dirs = sorted([p for p in artifact.glob("fold_*") if p.is_dir()])
    if not fold_dirs:
        raise ValueError(f"No fold_* directories found in {artifact}.")

    probs_by_fold: list[np.ndarray] = []
    fold_thresholds: list[float] = []

    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "clf_final_pass_mlp_embedding.pt"
        prep_path = fold_dir / "preprocess.json"
        metric_path = fold_dir / "metrics.json"
        if not ckpt_path.exists() or not prep_path.exists():
            raise ValueError(f"Fold artifact incomplete: {fold_dir}")

        preprocess = json.loads(prep_path.read_text(encoding="utf-8"))
        x_num, x_cat = transform_with_preprocess(df, preprocess)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = EmbeddingMLP(
            num_dim=int(ckpt["num_dim"]),
            cat_cardinalities=list(ckpt["cat_cardinalities"]),
            emb_dims=list(ckpt["emb_dims"]),
            hidden_dims=tuple(ckpt["hidden_dims"]),
            dropout=float(ckpt["dropout"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        fold_prob = predict_prob(model, x_num, x_cat, batch_size=batch_size, num_workers=0)
        probs_by_fold.append(fold_prob)

        fold_thresholds.append(_load_fold_threshold(metric_path))

    mean_prob = np.mean(np.stack(probs_by_fold, axis=0), axis=0)
    threshold_value = float(np.mean(fold_thresholds)) if threshold is None else float(threshold)
    threshold_value = float(np.clip(threshold_value, 0.0, 1.0))

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = mean_prob
    pred["pred_final_pass"] = mean_prob >= threshold_value

    if report_metrics:
        metrics = _collect_prediction_metrics(df, pred)
        if metrics:
            print("[surrogate][mlp-embedding-kfold] prediction metrics")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return pred


def predict_with_catboost(
    df: pd.DataFrame,
    artifact_dir: str | Path,
    threshold: float | None = None,
    report_metrics: bool = True,
) -> pd.DataFrame:
    feature_cols = PARAM_FEATURES + LAYOUT_FEATURES
    _check_required_columns(df, feature_cols)

    x = df[feature_cols].copy()
    for col in CAT_COLS:
        x[col] = x[col].astype(str)
    cat_indices = [x.columns.get_loc(col) for col in CAT_COLS]
    pool = Pool(x, cat_features=cat_indices)

    artifact = Path(artifact_dir)
    clf = joblib.load(artifact / "clf_final_pass_catboost.joblib")
    reg_models = {target: joblib.load(artifact / f"reg_{target}_catboost.joblib") for target in REG_TASKS}

    threshold = _load_threshold(artifact / "metrics.json") if threshold is None else threshold

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = clf.predict_proba(pool)[:, 1]
    pred["pred_final_pass"] = pred["pred_final_pass_prob"] >= threshold
    for target, model in reg_models.items():
        pred[f"pred_{target}"] = model.predict(pool)

    if report_metrics:
        metrics = _collect_prediction_metrics(df, pred)
        if metrics:
            metrics["classification"]["threshold"] = threshold
            print("[surrogate][catboost] prediction metrics")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return pred


def _prepare_lightgbm_features(
    df: pd.DataFrame, feature_cols: list[str], trained_cols: list[str]
) -> pd.DataFrame:
    x = df[feature_cols].copy()
    for col in CAT_COLS:
        x[col] = x[col].astype(str)
    x = pd.get_dummies(x, columns=CAT_COLS, dtype=float)
    return x.reindex(columns=trained_cols, fill_value=0.0)


def _check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Input dataset missing columns: {missing}")


def _load_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = metrics.get("classification", {}).get("threshold", 0.5)
    return float(np.clip(float(threshold), 0.0, 1.0))


def _collect_prediction_metrics(df: pd.DataFrame, pred: pd.DataFrame) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}

    cls_true_col = _pick_true_col(df, "final_pass")
    if cls_true_col is not None:
        y_true = df[cls_true_col].astype(int).to_numpy()
        y_prob = pred["pred_final_pass_prob"].astype(float).to_numpy()
        y_pred = pred["pred_final_pass"].astype(int).to_numpy()

        cls_metrics: dict[str, float] = {
            "samples": float(len(y_true)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            cls_metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            cls_metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        report["classification"] = cls_metrics

    reg_metrics: dict[str, float] = {}
    for target in REG_TASKS:
        true_col = _pick_true_col(df, target)
        pred_col = f"pred_{target}"
        if true_col is None or pred_col not in pred.columns:
            continue

        y_true = df[true_col].astype(float).to_numpy()
        y_pred = pred[pred_col].astype(float).to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        # reg_metrics[f"{target}.samples"] = float(len(yt))
        reg_metrics[f"{target}.mae"] = float(mean_absolute_error(yt, yp))
        reg_metrics[f"{target}.rmse"] = float(np.sqrt(np.mean((yp - yt) ** 2)))
        reg_metrics[f"{target}.r2"] = float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")

    if reg_metrics:
        report["regression"] = reg_metrics

    return report


def _pick_true_col(df: pd.DataFrame, target: str) -> str | None:
    if f"{target}_true" in df.columns:
        return f"{target}_true"
    if target in df.columns:
        return target
    return None


def _load_fold_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    t = raw.get("threshold", 0.5)
    return float(np.clip(float(t), 0.0, 1.0))
