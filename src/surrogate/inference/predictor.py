from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from src.surrogate.training.lightgbm_baseline import LAYOUT_FEATURES, PARAM_FEATURES

CAT_COLS = ["conc_bot", "site_class", "intensity", "seismic_group"]
REG_TASKS = ["max_drift_ratio", "torsion_ratio", "material_steel_kg"]


def predict_with_lightgbm(df: pd.DataFrame, artifact_dir: str | Path) -> pd.DataFrame:
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

    threshold = _load_threshold(artifact / "metrics.json")

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = clf.predict_proba(x)[:, 1]
    pred["pred_final_pass"] = pred["pred_final_pass_prob"] >= threshold
    for target, model in reg_models.items():
        pred[f"pred_{target}"] = model.predict(x)
    return pred


def predict_with_catboost(df: pd.DataFrame, artifact_dir: str | Path) -> pd.DataFrame:
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

    threshold = _load_threshold(artifact / "metrics.json")

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = clf.predict_proba(pool)[:, 1]
    pred["pred_final_pass"] = pred["pred_final_pass_prob"] >= threshold
    for target, model in reg_models.items():
        pred[f"pred_{target}"] = model.predict(pool)
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
