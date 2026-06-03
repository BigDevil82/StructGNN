from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.surrogate.features.consts import LAYOUT_FEATURES, PARAM_FEATURES

CLASS_TASK = "final_pass"
REG_TASKS = [
    "max_drift_ratio",
    "torsion_ratio",
    "material_steel_kg",
    "min_shear_weight_ratio",
    "margin_drift",
    "margin_torsion",
    "margin_shear",
    "period_ratio",
]


@dataclass(frozen=True)
class LightGBMBaselineConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    output_dir: str = r"data\parametric\surrogate_dataset\baseline_lightgbm"
    seed: int = 42
    n_estimators: int = 800
    learning_rate: float = 0.05
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50


def run_lightgbm_baseline(cfg: LightGBMBaselineConfig) -> dict[str, object]:
    df = pd.read_parquet(cfg.dataset_path)
    required = ["split", CLASS_TASK] + PARAM_FEATURES + LAYOUT_FEATURES + REG_TASKS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test split is empty. Please regenerate split files.")

    feature_cols = PARAM_FEATURES + LAYOUT_FEATURES
    cat_cols = ["conc_bot", "site_class", "intensity", "seismic_group"]

    x_train, x_val, x_test = _encode_features(train_df, val_df, test_df, feature_cols, cat_cols)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "dataset_path": cfg.dataset_path,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "feature_count": int(x_train.shape[1]),
        "config": asdict(cfg),
    }

    clf_metrics, threshold = _train_classifier(
        cfg, x_train, x_val, x_test, train_df, val_df, test_df, out_dir
    )
    metrics["classification"] = clf_metrics
    metrics["classification"]["threshold"] = threshold

    reg_metrics: dict[str, dict[str, float]] = {}
    for target in REG_TASKS:
        print(f"Training regressor for target: {target}...")
        reg_metrics[target] = _train_regressor(
            cfg, target, x_train, x_val, x_test, train_df, val_df, test_df, out_dir
        )
    metrics["regression"] = reg_metrics

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")

    feature_cols_path = out_dir / "feature_columns.json"
    feature_cols_path.write_text(
        json.dumps(list(x_train.columns), ensure_ascii=True, indent=2), encoding="utf-8"
    )

    return metrics


def _encode_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].astype(str)
        x_val[col] = x_val[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    x_train = pd.get_dummies(x_train, columns=cat_cols, dtype=float)
    x_val = pd.get_dummies(x_val, columns=cat_cols, dtype=float).reindex(
        columns=x_train.columns, fill_value=0.0
    )
    x_test = pd.get_dummies(x_test, columns=cat_cols, dtype=float).reindex(
        columns=x_train.columns, fill_value=0.0
    )

    return x_train, x_val, x_test


def _train_classifier(
    cfg: LightGBMBaselineConfig,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[dict[str, float], float]:
    y_train = train_df[CLASS_TASK].astype(int).to_numpy()
    y_val = val_df[CLASS_TASK].astype(int).to_numpy()
    y_test = test_df[CLASS_TASK].astype(int).to_numpy()

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        class_weight="balanced",
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )

    y_val_prob = model.predict_proba(x_val)[:, 1]
    threshold = _find_best_f1_threshold(y_val, y_val_prob)

    y_test_prob = model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_test_prob)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob)),
        "f1": float(f1_score(y_test, y_test_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
        "brier": float(np.mean((y_test_prob - y_test) ** 2)),
    }

    model_path = out_dir / "clf_final_pass_lightgbm.joblib"
    joblib.dump(model, model_path)

    imp = pd.DataFrame({"feature": x_train.columns, "importance": model.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    imp.to_csv(out_dir / "clf_final_pass_feature_importance.csv", index=False)
    return metrics, threshold


def _train_regressor(
    cfg: LightGBMBaselineConfig,
    target: str,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, float]:
    train_mask = np.isfinite(train_df[target].to_numpy(dtype=float))
    val_mask = np.isfinite(val_df[target].to_numpy(dtype=float))
    test_mask = np.isfinite(test_df[target].to_numpy(dtype=float))

    x_train_reg = x_train.loc[train_mask]
    x_val_reg = x_val.loc[val_mask]
    x_test_reg = x_test.loc[test_mask]

    y_train = train_df.loc[train_mask, target].to_numpy(dtype=float)
    y_val = val_df.loc[val_mask, target].to_numpy(dtype=float)
    y_test = test_df.loc[test_mask, target].to_numpy(dtype=float)

    if len(y_train) == 0 or len(y_val) == 0 or len(y_test) == 0:
        raise ValueError(f"Target {target} has no valid finite samples in train/val/test.")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(
        x_train_reg,
        y_train,
        eval_set=[(x_val_reg, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )

    y_pred = model.predict(x_test_reg)
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }

    model_path = out_dir / f"reg_{target}_lightgbm.joblib"
    joblib.dump(model, model_path)

    imp = pd.DataFrame(
        {"feature": x_train_reg.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    imp.to_csv(out_dir / f"reg_{target}_feature_importance.csv", index=False)
    return metrics


def _find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr
