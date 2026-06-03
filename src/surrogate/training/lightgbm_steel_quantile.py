from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.surrogate.features.consts import LAYOUT_FEATURES, PARAM_FEATURES

TARGET = "material_steel_kg"
CAT_COLS = ["conc_bot", "site_class", "intensity", "seismic_group"]


@dataclass(frozen=True)
class SteelQuantileConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    output_dir: str = r"data\parametric\surrogate_dataset\steel_quantile_lightgbm"
    seed: int = 42
    n_estimators: int = 1200
    learning_rate: float = 0.04
    num_leaves: int = 63
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    upper_alpha: float = 0.95
    early_stopping_rounds: int = 80


def train_steel_quantile_model(cfg: SteelQuantileConfig) -> dict[str, object]:
    df = pd.read_parquet(cfg.dataset_path)
    feature_cols = PARAM_FEATURES + LAYOUT_FEATURES
    required = ["split", TARGET] + feature_cols
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test split is empty")

    x_train, x_val, x_test, trained_cols = _encode(train_df, val_df, test_df, feature_cols)
    y_train = train_df[TARGET].astype(float).to_numpy()
    y_val = val_df[TARGET].astype(float).to_numpy()
    y_test = test_df[TARGET].astype(float).to_numpy()

    mean_model = _fit_mean_model(cfg, x_train, y_train, x_val, y_val)
    upper_model = _fit_quantile_model(cfg, cfg.upper_alpha, x_train, y_train, x_val, y_val)

    # val_mean = mean_model.predict(x_val)
    val_upper_raw = upper_model.predict(x_val)
    cal_margin = _calibrate_upper_margin(y_val, val_upper_raw, cfg.upper_alpha)

    test_mean = mean_model.predict(x_test)
    test_upper_raw = upper_model.predict(x_test)
    test_upper = test_upper_raw + cal_margin

    metrics = {
        "config": asdict(cfg),
        "feature_count": int(len(trained_cols)),
        "calibration": {
            "upper_alpha": float(cfg.upper_alpha),
            "upper_margin_kg": float(cal_margin),
        },
        "test": {
            "mean_mae": float(mean_absolute_error(y_test, test_mean)),
            "mean_rmse": float(np.sqrt(mean_squared_error(y_test, test_mean))),
            "mean_r2": float(r2_score(y_test, test_mean)),
            "upper_coverage": float(np.mean(y_test <= test_upper)),
            "upper_mean_overage_kg": float(np.mean(test_upper - y_test)),
            "upper_p95_overage_kg": float(np.quantile(test_upper - y_test, 0.95)),
            "upper_under_count": int(np.sum(y_test > test_upper)),
        },
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mean_model, out_dir / "steel_mean_lightgbm.joblib")
    joblib.dump(upper_model, out_dir / "steel_upper_lightgbm.joblib")
    (out_dir / "feature_columns.json").write_text(
        json.dumps(trained_cols, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def _fit_mean_model(cfg: SteelQuantileConfig, x_train, y_train, x_val, y_val):
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
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    return model


def _fit_quantile_model(cfg: SteelQuantileConfig, alpha: float, x_train, y_train, x_val, y_val):
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="quantile",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )
    return model


def _encode(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    for col in CAT_COLS:
        x_train[col] = x_train[col].astype(str)
        x_val[col] = x_val[col].astype(str)
        x_test[col] = x_test[col].astype(str)

    x_train = pd.get_dummies(x_train, columns=CAT_COLS, dtype=float)
    trained_cols = list(x_train.columns)
    x_val = pd.get_dummies(x_val, columns=CAT_COLS, dtype=float).reindex(columns=trained_cols, fill_value=0.0)
    x_test = pd.get_dummies(x_test, columns=CAT_COLS, dtype=float).reindex(
        columns=trained_cols, fill_value=0.0
    )
    return x_train, x_val, x_test, trained_cols


def _calibrate_upper_margin(y_true: np.ndarray, y_upper: np.ndarray, coverage: float) -> float:
    residual = y_true - y_upper
    return float(max(0.0, np.quantile(residual, coverage)))
