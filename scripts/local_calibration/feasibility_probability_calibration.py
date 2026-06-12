from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.surrogate.features.consts import PARAM_FEATURES

TARGET = "final_pass_true"
PROB = "pred_final_pass_prob"


@dataclass(frozen=True)
class LocalMethod:
    name: str
    fit: Callable[[pd.DataFrame, np.ndarray, np.ndarray, int], object]
    predict: Callable[[object, pd.DataFrame, np.ndarray], np.ndarray]


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(Path(args.dataset_path), Path(args.prediction_path), args.input_mode)
    layouts = select_layouts(df, args.layouts, args.max_layouts, args.seed)
    calib_sizes = parse_int_list(args.calib_sizes)
    method_names = parse_str_list(args.methods)
    methods = method_specs()
    missing = sorted(set(method_names) - set(methods))
    if missing:
        raise ValueError(f"Unknown methods: {missing}. Available methods: {sorted(methods)}")

    rows = []
    for layout_id in layouts:
        g = df[df["layout_id"].astype(str) == layout_id].copy().reset_index(drop=True)
        max_calib = min(max(calib_sizes), max(1, len(g) - args.min_eval_samples))
        if max_calib <= 0:
            continue
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.seed + repeat * 1009 + stable_hash(layout_id))
            order = rng.permutation(len(g))
            for calib_n in calib_sizes:
                if calib_n > max_calib:
                    continue
                calib = g.iloc[order[:calib_n]].copy()
                eval_df = g.iloc[order[calib_n:]].copy()
                if len(eval_df) < args.min_eval_samples:
                    continue

                y_calib = calib[TARGET].astype(int).to_numpy()
                p_calib = calib[PROB].astype(float).to_numpy()
                y_eval = eval_df[TARGET].astype(int).to_numpy()
                p_eval = eval_df[PROB].astype(float).to_numpy()
                x_calib = feature_frame(calib, args.input_mode)
                x_eval = feature_frame(eval_df, args.input_mode)

                base_thr = best_f1_threshold(y_calib, p_calib)
                base = evaluate(
                    y_eval,
                    p_eval,
                    base_thr,
                    args.screening_target_recall,
                    args.screening_threshold_scale,
                    y_calib,
                    p_calib,
                )
                rows.append(result_row(layout_id, repeat, calib_n, "global_local_threshold", base, base))

                fixed = evaluate(
                    y_eval,
                    p_eval,
                    0.5,
                    args.screening_target_recall,
                    args.screening_threshold_scale,
                    y_calib,
                    p_calib,
                )
                rows.append(result_row(layout_id, repeat, calib_n, "global_0p5", fixed, base))

                for name in method_names:
                    method = methods[name]
                    model = method.fit(x_calib, y_calib, p_calib, args.seed + repeat)
                    p_adj_calib = method.predict(model, x_calib, p_calib)
                    p_adj_eval = method.predict(model, x_eval, p_eval)
                    thr = best_f1_threshold(y_calib, p_adj_calib)
                    stat = evaluate(
                        y_eval,
                        p_adj_eval,
                        thr,
                        args.screening_target_recall,
                        args.screening_threshold_scale,
                        y_calib,
                        p_adj_calib,
                    )
                    rows.append(result_row(layout_id, repeat, calib_n, name, stat, base))

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No calibration result was produced. Check layouts and calibration sizes.")

    result.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    summarize(result).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    best_by_layout(result).to_csv(out_dir / "best_by_layout.csv", index=False, encoding="utf-8-sig")
    write_report(result, args, out_dir / "report.md")
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[feas-calib] rows={len(result)} layouts={len(layouts)} out={out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate layout-local calibration for feasibility probabilities.")
    p.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    p.add_argument(
        "--prediction-path",
        default=r"data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1\predictions_test.parquet",
    )
    p.add_argument("--output-dir", default=r"outputs\result\local_calibration\feasibility_probability")
    p.add_argument("--layouts", default="all", help="Comma-separated layout ids, or all.")
    p.add_argument("--max-layouts", type=int, default=0)
    p.add_argument("--calib-sizes", default="25,50,100,200,500,1000")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-eval-samples", type=int, default=200)
    p.add_argument(
        "--methods",
        default="platt,isotonic,logistic,residual_ridge,residual_rf,residual_lgbm,vote",
    )
    p.add_argument("--input-mode", choices=["prob", "param", "prob_param"], default="prob_param")
    p.add_argument("--screening-target-recall", type=float, default=0.995)
    p.add_argument("--safe-screen-recall-floor", type=float, default=0.99)
    p.add_argument(
        "--screening-threshold-scale",
        type=float,
        default=1.0,
        help="Multiply local screening threshold by this factor. Values below 1 are more conservative.",
    )
    return p.parse_args()


def load_data(dataset_path: Path, prediction_path: Path, input_mode: str) -> pd.DataFrame:
    pred = normalize_prediction_cols(pd.read_parquet(prediction_path))
    need_cols = ["layout_id", "sample_id"]
    if input_mode in {"param", "prob_param"}:
        need_cols += PARAM_FEATURES
    data = pd.read_parquet(dataset_path, columns=need_cols)
    data["layout_id"] = data["layout_id"].astype(str)
    pred["layout_id"] = pred["layout_id"].astype(str)
    out = pred.merge(data, on=["layout_id", "sample_id"], how="inner", validate="one_to_one")
    out = out.dropna(subset=[TARGET, PROB]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Prediction and dataset rows did not match.")
    out[PROB] = clip_prob(out[PROB].to_numpy())
    return out


def normalize_prediction_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if TARGET not in df.columns:
        for col in ["final_pass", "feasible", "y_true", "true", "target"]:
            if col in df.columns:
                rename[col] = TARGET
                break
    if PROB not in df.columns:
        for col in ["prob", "y_prob", "pred_prob", "pred_feasible_prob", "pred_final_pass_probability"]:
            if col in df.columns:
                rename[col] = PROB
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["layout_id", "sample_id", TARGET, PROB] if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")
    return df[["layout_id", "sample_id", TARGET, PROB]].copy()


def select_layouts(df: pd.DataFrame, layouts: str, max_layouts: int, seed: int) -> list[str]:
    available = sorted(df["layout_id"].astype(str).unique().tolist())
    if layouts.strip().lower() != "all":
        wanted = parse_str_list(layouts)
        missing = sorted(set(wanted) - set(available))
        if missing:
            raise ValueError(f"Layouts not found in merged data: {missing}")
        available = wanted
    if max_layouts and max_layouts < len(available):
        rng = np.random.default_rng(seed)
        available = sorted(rng.choice(available, size=max_layouts, replace=False).tolist())
    return available


def method_specs() -> dict[str, LocalMethod]:
    specs = {
        "platt": LocalMethod("platt", fit_platt, predict_platt),
        "isotonic": LocalMethod("isotonic", fit_isotonic, predict_isotonic),
        "logistic": LocalMethod("logistic", fit_logistic, predict_classifier),
        "residual_ridge": LocalMethod("residual_ridge", fit_residual_ridge, predict_residual),
        "residual_rf": LocalMethod("residual_rf", fit_residual_rf, predict_residual),
        "knn": LocalMethod("knn", fit_knn, predict_classifier),
    }
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        specs["residual_lgbm"] = LocalMethod(
            "residual_lgbm",
            lambda x, y, p, seed: fit_residual_model(
                x,
                y,
                p,
                LGBMRegressor(
                    objective="regression",
                    n_estimators=200,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=seed,
                    verbose=-1,
                ),
            ),
            predict_residual,
        )
        specs["vote"] = LocalMethod(
            "vote",
            lambda x, y, p, seed: fit_vote(x, y, p, seed, LGBMClassifier),
            predict_vote,
        )
    except ImportError:
        specs["vote"] = LocalMethod("vote", lambda x, y, p, seed: fit_vote(x, y, p, seed, None), predict_vote)
    return specs


def fit_platt(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    if len(np.unique(y)) < 2:
        return {"fallback": "identity"}
    model = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
    model.fit(logit_feature(p), y)
    return model


def predict_platt(model: object, x: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return clip_prob(p)
    return clip_prob(model.predict_proba(logit_feature(p))[:, 1])


def fit_isotonic(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    if len(np.unique(y)) < 2:
        return {"fallback": "identity"}
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p, y)
    return model


def predict_isotonic(model: object, x: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return clip_prob(p)
    return clip_prob(model.predict(p))


def fit_logistic(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    if len(np.unique(y)) < 2:
        return {"fallback": "identity"}
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
    return build_classifier_pipeline(x, model).fit(x, y)


def fit_knn(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    if len(np.unique(y)) < 2:
        return {"fallback": "identity"}
    k = max(3, min(15, len(y) // 3))
    model = KNeighborsClassifier(n_neighbors=k, weights="distance")
    return build_classifier_pipeline(x, model).fit(x, y)


def predict_classifier(model: object, x: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict):
        return clip_prob(p)
    return clip_prob(model.predict_proba(x)[:, 1])


def fit_residual_ridge(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    return fit_residual_model(x, y, p, Ridge(alpha=10.0, random_state=seed))


def fit_residual_rf(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int) -> object:
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )
    return fit_residual_model(x, y, p, model)


def fit_residual_model(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, model: object) -> object:
    pipe = build_regressor_pipeline(x, model)
    pipe.fit(x, y.astype(float) - p.astype(float))
    return pipe


def predict_residual(model: object, x: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    return clip_prob(p + model.predict(x))


def fit_vote(x: pd.DataFrame, y: np.ndarray, p: np.ndarray, seed: int, lgbm_cls: object | None) -> object:
    if len(np.unique(y)) < 2:
        return {"fallback": "identity"}
    models = [
        build_classifier_pipeline(
            x, LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=seed)
        ),
        build_classifier_pipeline(
            x,
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            )
        ),
        build_classifier_pipeline(
            x, KNeighborsClassifier(n_neighbors=max(3, min(15, len(y) // 3)), weights="distance")
        ),
    ]
    if lgbm_cls is not None:
        models.append(
            build_classifier_pipeline(
                x,
                lgbm_cls(
                    objective="binary",
                    n_estimators=200,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=5,
                    class_weight="balanced",
                    random_state=seed,
                    verbose=-1,
                )
            )
        )
    fitted = []
    for model in models:
        try:
            fitted.append(clone(model).fit(x, y))
        except Exception:
            pass
    if not fitted:
        return {"fallback": "identity"}
    return {"models": fitted}


def predict_vote(model: object, x: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    if isinstance(model, dict) and "models" not in model:
        return clip_prob(p)
    probs = [clip_prob(p)]
    probs += [clip_prob(m.predict_proba(x)[:, 1]) for m in model["models"]]
    return clip_prob(np.mean(probs, axis=0))


def build_classifier_pipeline(x: pd.DataFrame, model: object) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(x, scaled=True)), ("model", model)])


def build_regressor_pipeline(x: pd.DataFrame, model: object) -> Pipeline:
    scaled = isinstance(model, Ridge)
    return Pipeline([("pre", build_preprocessor(x, scaled=scaled)), ("model", model)])


def build_preprocessor(x: pd.DataFrame, *, scaled: bool) -> ColumnTransformer:
    numeric = [c for c in x.columns if is_numeric_dtype(x[c])]
    categorical = [c for c in x.columns if not is_numeric_dtype(x[c])]
    return ColumnTransformer(make_transformers(numeric, categorical, scaled), sparse_threshold=0.0)


def make_transformers(numeric: list[str], categorical: list[str], scaled: bool) -> list[tuple[str, object, list[str]]]:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        num_steps.append(("scaler", StandardScaler()))
    return [
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())]), categorical),
    ]


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def feature_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = []
    if mode in {"prob", "prob_param"}:
        cols += [PROB]
    if mode in {"param", "prob_param"}:
        cols += PARAM_FEATURES
    x = df[cols].copy()
    if PROB in x.columns:
        p = clip_prob(x[PROB].astype(float).to_numpy())
        x["logit_" + PROB] = safe_logit(p)
    return x


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    target_recall: float,
    threshold_scale: float,
    y_calib: np.ndarray,
    p_calib: np.ndarray,
) -> dict[str, float]:
    p = clip_prob(p)
    pred = (p >= threshold).astype(int)
    raw_screen_thr = screening_threshold(y_calib, p_calib, target_recall)
    screen_thr = max(0.0, min(1.0, raw_screen_thr * threshold_scale))
    screen = p < screen_thr
    return {
        "samples": float(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "threshold": float(threshold),
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": safe_balanced_accuracy(y, pred),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": safe_log_loss(y, p),
        "ece": expected_calibration_error(y, p, bins=10),
        "screen_threshold": float(screen_thr),
        "screen_threshold_raw": float(raw_screen_thr),
        "screen_reject_rate": float(screen.mean()),
        "screen_false_reject_rate": float(np.mean(screen & (y == 1))),
        "screen_recall": float(np.sum((~screen) & (y == 1)) / max(1, np.sum(y == 1))),
    }


def result_row(
    layout_id: str,
    repeat: int,
    calib_n: int,
    method: str,
    stat: dict[str, float],
    base: dict[str, float],
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "layout_id": layout_id,
        "repeat": repeat,
        "calib_n": calib_n,
        "method": method,
    }
    row.update(stat)
    for key in ["brier", "ece", "f1", "pr_auc", "screen_reject_rate", "screen_false_reject_rate"]:
        row[f"{key}_delta"] = stat[key] - base[key]
    row["brier_improve_pct"] = (base["brier"] - stat["brier"]) / max(base["brier"], 1.0e-9)
    row["ece_improve_pct"] = (base["ece"] - stat["ece"]) / max(base["ece"], 1.0e-9)
    return row


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "samples",
        "positive_rate",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "balanced_accuracy",
        "brier",
        "log_loss",
        "ece",
        "screen_reject_rate",
        "screen_false_reject_rate",
        "screen_recall",
        "brier_improve_pct",
        "ece_improve_pct",
    ]
    return (
        result.groupby(["calib_n", "method"], as_index=False)[cols]
        .mean()
        .sort_values(["calib_n", "brier"])
        .reset_index(drop=True)
    )


def best_by_layout(result: pd.DataFrame) -> pd.DataFrame:
    calibrated = result[~result["method"].str.startswith("global")].copy()
    idx = calibrated.groupby(["layout_id", "calib_n"])["brier"].idxmin()
    return calibrated.loc[idx].sort_values(["layout_id", "calib_n"]).reset_index(drop=True)


def write_report(result: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    summary = summarize(result)
    calibrated = summary[~summary["method"].str.startswith("global")].copy()
    best = calibrated.sort_values(["calib_n", "brier"]).groupby("calib_n", as_index=False).first()
    safe = calibrated[calibrated["screen_recall"] >= float(args.safe_screen_recall_floor)].copy()
    if not safe.empty:
        safe = safe.sort_values(["calib_n", "screen_reject_rate"], ascending=[True, False])
        safe = safe.groupby("calib_n", as_index=False).first()
    lines = [
        "# Feasibility Probability Local Calibration",
        "",
        f"- prediction_path: `{args.prediction_path}`",
        f"- dataset_path: `{args.dataset_path}`",
        f"- input_mode: `{args.input_mode}`",
        f"- repeats: `{args.repeats}`",
        f"- screening_target_recall: `{args.screening_target_recall}`",
        f"- screening_threshold_scale: `{args.screening_threshold_scale}`",
        f"- safe_screen_recall_floor: `{args.safe_screen_recall_floor}`",
        "",
        "## Best Mean Result By Calibration Size",
        "",
        "| calib_n | method | Brier | ECE | PR-AUC | F1 | screen reject | screen recall |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| {int(row['calib_n'])} | {row['method']} | {row['brier']:.5f} | "
            f"{row['ece']:.5f} | {row['pr_auc']:.4f} | {row['f1']:.4f} | "
            f"{row['screen_reject_rate']:.1%} | {row['screen_recall']:.4f} |"
        )
    lines += [
        "",
        "## Best Safe Screening Candidate",
        "",
        "| calib_n | method | screen reject | screen recall | Brier | ECE | F1 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    if safe.empty:
        lines.append("| - | - | - | - | - | - | - |")
    else:
        for row in safe.to_dict("records"):
            lines.append(
                f"| {int(row['calib_n'])} | {row['method']} | {row['screen_reject_rate']:.1%} | "
                f"{row['screen_recall']:.4f} | {row['brier']:.5f} | {row['ece']:.5f} | {row['f1']:.4f} |"
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "`global_local_threshold` keeps the original probability but re-selects the decision threshold from local calibration samples.",
        "Platt and isotonic calibrate probability using only the original global probability.",
        "Logistic, residual and vote methods use the original probability plus design variables when `input_mode=prob_param`.",
        "For optimization screening, the most relevant columns are `screen_reject_rate` and `screen_recall`; for confidence quality, inspect `brier` and `ece`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_prob(p)
    if len(np.unique(y)) < 2:
        return 0.5
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, 101)))
    best_thr = 0.5
    best = -1.0
    for thr in qs:
        score = f1_score(y, (p >= thr).astype(int), zero_division=0)
        if score > best:
            best = float(score)
            best_thr = float(thr)
    return best_thr


def screening_threshold(y: np.ndarray, p: np.ndarray, target_recall: float) -> float:
    p = clip_prob(p)
    pos = p[y.astype(int) == 1]
    if len(pos) == 0:
        return 0.0
    miss_rate = max(0.0, min(1.0, 1.0 - target_recall))
    return float(np.quantile(pos, miss_rate))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int) -> float:
    p = clip_prob(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return ece if total else 0.0


def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    if np.sum(y == 1) == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def safe_balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(balanced_accuracy_score(y, pred))


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    labels = [0, 1]
    return float(log_loss(y, clip_prob(p), labels=labels))


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)


def safe_logit(p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return np.log(p / (1.0 - p))


def logit_feature(p: np.ndarray) -> np.ndarray:
    return safe_logit(p).reshape(-1, 1)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def stable_hash(text: str) -> int:
    val = 0
    for ch in text:
        val = (val * 131 + ord(ch)) % 1_000_000_007
    return val


if __name__ == "__main__":
    main()
