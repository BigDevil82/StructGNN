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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.surrogate.features.consts import PARAM_FEATURES

TARGET = "steel_true_kg"
PRED = "steel_pred_kg"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    build: Callable[[int], object]
    needs_scaling: bool = True
    max_train: int | None = None


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        dataset_path=Path(args.dataset_path),
        prediction_path=Path(args.prediction_path),
        input_mode=args.input_mode,
    )
    layouts = select_layouts(df, args.layouts, args.max_layouts, args.seed)
    calib_sizes = parse_int_list(args.calib_sizes)
    model_names = parse_str_list(args.models)
    specs = model_specs(args.max_gpr_train)
    missing_models = sorted(set(model_names) - set(specs))
    if missing_models:
        raise ValueError(f"Unavailable calibration models: {missing_models}. Installed models: {sorted(specs)}")

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

                base = evaluate(eval_df[TARGET].to_numpy(), eval_df[PRED].to_numpy())
                rows.append(
                    result_row(
                        layout_id,
                        repeat,
                        calib_n,
                        "global",
                        base,
                        base,
                        actual_train_n=0,
                    )
                )

                x_calib = feature_frame(calib, args.input_mode)
                x_eval = feature_frame(eval_df, args.input_mode)
                y_res = residual_target(calib, args.target_space)
                for name in model_names:
                    spec = specs[name]
                    train_idx = capped_train_index(len(calib), spec.max_train, rng)
                    model = build_pipeline(spec, x_calib, args.seed + repeat)
                    model.fit(x_calib.iloc[train_idx], y_res[train_idx])
                    res_pred = model.predict(x_eval)
                    y_pred = apply_residual(eval_df[PRED].to_numpy(), res_pred, args.target_space)
                    stat = evaluate(eval_df[TARGET].to_numpy(), y_pred)
                    rows.append(
                        result_row(
                            layout_id,
                            repeat,
                            calib_n,
                            name,
                            stat,
                            base,
                            actual_train_n=len(train_idx),
                        )
                    )

    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No calibration result was produced. Check layouts and calibration sizes.")

    result.to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    summarize(result).to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    best_by_layout(result).to_csv(out_dir / "best_by_layout.csv", index=False, encoding="utf-8-sig")
    write_report(result, args, out_dir / "report.md")
    (out_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"[local-calib] rows={len(result)} layouts={len(layouts)} out={out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate layout-local residual calibration for steel regression.")
    p.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
        help="Parquet with layout_id, sample_id and parameter columns.",
    )
    p.add_argument(
        "--prediction-path",
        default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\predictions_test.parquet",
        help="Global steel model predictions with layout_id, sample_id, steel_true_kg, steel_pred_kg.",
    )
    p.add_argument("--output-dir", default=r"outputs\result\local_calibration\steel_residual")
    p.add_argument("--layouts", default="all", help="Comma-separated layout ids, or all.")
    p.add_argument("--max-layouts", type=int, default=0, help="Randomly limit layouts for quick runs; 0 means no limit.")
    p.add_argument("--calib-sizes", default="25,50,100,200,500,1000")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-eval-samples", type=int, default=200)
    p.add_argument(
        "--models",
        default="ridge,rf,lgbm,gpr,knn,mlp",
        help="Comma-separated subset of ridge,rf,lgbm,gpr,knn,mlp.",
    )
    p.add_argument("--input-mode", choices=["pred", "param", "pred_param"], default="pred_param")
    p.add_argument("--target-space", choices=["kg", "log"], default="kg")
    p.add_argument("--max-gpr-train", type=int, default=300)
    return p.parse_args()


def load_data(dataset_path: Path, prediction_path: Path, input_mode: str) -> pd.DataFrame:
    pred = pd.read_parquet(prediction_path).copy()
    pred = normalize_prediction_cols(pred)
    need_cols = ["layout_id", "sample_id"]
    if input_mode in {"param", "pred_param"}:
        need_cols += PARAM_FEATURES
    data = pd.read_parquet(dataset_path, columns=need_cols)
    data["layout_id"] = data["layout_id"].astype(str)
    pred["layout_id"] = pred["layout_id"].astype(str)
    out = pred.merge(data, on=["layout_id", "sample_id"], how="inner", validate="one_to_one")
    out = out.dropna(subset=[TARGET, PRED]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("Prediction and dataset rows did not match.")
    return out


def normalize_prediction_cols(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if TARGET not in df.columns:
        for col in ["material_steel_kg", "y_true", "true", "target"]:
            if col in df.columns:
                rename[col] = TARGET
                break
    if PRED not in df.columns:
        for col in ["pred_material_steel_kg", "y_pred", "pred", "prediction"]:
            if col in df.columns:
                rename[col] = PRED
                break
    df = df.rename(columns=rename)
    missing = [c for c in ["layout_id", "sample_id", TARGET, PRED] if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file missing columns: {missing}")
    return df[["layout_id", "sample_id", TARGET, PRED]].copy()


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


def model_specs(max_gpr_train: int) -> dict[str, ModelSpec]:
    specs = {
        "ridge": ModelSpec("ridge", lambda seed: Ridge(alpha=10.0, random_state=seed), needs_scaling=True),
        "rf": ModelSpec(
            "rf",
            lambda seed: RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=seed,
                n_jobs=1,
            ),
            needs_scaling=False,
        ),
        "gpr": ModelSpec(
            "gpr",
            lambda seed: GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                alpha=1.0e-6,
                normalize_y=True,
                random_state=seed,
            ),
            needs_scaling=True,
            max_train=max_gpr_train,
        ),
        "knn": ModelSpec(
            "knn",
            lambda seed: KNeighborsRegressor(n_neighbors=15, weights="distance", p=2),
            needs_scaling=True,
        ),
        "mlp": ModelSpec(
            "mlp",
            lambda seed: MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                alpha=1.0e-4,
                learning_rate_init=1.0e-3,
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            ),
            needs_scaling=True,
        ),
    }
    try:
        from lightgbm import LGBMRegressor

        specs["lgbm"] = ModelSpec(
            "lgbm",
            lambda seed: LGBMRegressor(
                objective="regression",
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=10,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                verbose=-1,
            ),
            needs_scaling=False,
        )
    except ImportError:
        pass
    return specs


def build_pipeline(spec: ModelSpec, x: pd.DataFrame, seed: int) -> Pipeline:
    numeric = [c for c in x.columns if not is_categorical(x[c])]
    categorical = [c for c in x.columns if is_categorical(x[c])]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if spec.needs_scaling:
        num_steps.append(("scaler", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", make_onehot())]),
                categorical,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return Pipeline([("pre", pre), ("model", spec.build(seed))])


def make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def is_categorical(s: pd.Series) -> bool:
    return not is_numeric_dtype(s)


def feature_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    cols = []
    if mode in {"pred", "pred_param"}:
        cols += [PRED]
    if mode in {"param", "pred_param"}:
        cols += PARAM_FEATURES
    x = df[cols].copy()
    if PRED in x.columns:
        x["log_" + PRED] = np.log1p(np.maximum(x[PRED].astype(float), 0.0))
    return x


def residual_target(df: pd.DataFrame, space: str) -> np.ndarray:
    y = df[TARGET].to_numpy(dtype=np.float64)
    p = df[PRED].to_numpy(dtype=np.float64)
    if space == "kg":
        return y - p
    return np.log1p(np.maximum(y, 0.0)) - np.log1p(np.maximum(p, 0.0))


def apply_residual(pred: np.ndarray, residual: np.ndarray, space: str) -> np.ndarray:
    pred = pred.astype(np.float64)
    residual = residual.astype(np.float64)
    if space == "kg":
        return np.maximum(pred + residual, 0.0)
    return np.maximum(np.expm1(np.log1p(np.maximum(pred, 0.0)) + residual), 0.0)


def capped_train_index(n: int, max_train: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_train is None or n <= max_train:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_train, replace=False))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float64)
    y_pred = np.maximum(y_pred.astype(np.float64), 0.0)
    err = y_pred - y_true
    return {
        "samples": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(err) / np.maximum(y_true, 1.0))),
        "bias": float(err.mean()),
        "p90_abs_error": float(np.quantile(np.abs(err), 0.9)),
    }


def result_row(
    layout_id: str,
    repeat: int,
    calib_n: int,
    model: str,
    stat: dict[str, float],
    base: dict[str, float],
    *,
    actual_train_n: int,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "layout_id": layout_id,
        "repeat": repeat,
        "calib_n": calib_n,
        "model": model,
        "actual_train_n": actual_train_n,
    }
    for k, v in stat.items():
        row[k] = v
    row["mae_improve_pct"] = (base["mae"] - stat["mae"]) / max(base["mae"], 1.0e-9)
    row["rmse_improve_pct"] = (base["rmse"] - stat["rmse"]) / max(base["rmse"], 1.0e-9)
    row["mape_improve_pct"] = (base["mape"] - stat["mape"]) / max(base["mape"], 1.0e-9)
    return row


def summarize(result: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "samples",
        "mae",
        "rmse",
        "r2",
        "mape",
        "bias",
        "p90_abs_error",
        "mae_improve_pct",
        "rmse_improve_pct",
        "mape_improve_pct",
    ]
    return (
        result.groupby(["calib_n", "model"], as_index=False)[cols]
        .mean()
        .sort_values(["calib_n", "mae"])
        .reset_index(drop=True)
    )


def best_by_layout(result: pd.DataFrame) -> pd.DataFrame:
    calibrated = result[result["model"] != "global"].copy()
    idx = calibrated.groupby(["layout_id", "calib_n"])["mae"].idxmin()
    return calibrated.loc[idx].sort_values(["layout_id", "calib_n"]).reset_index(drop=True)


def write_report(result: pd.DataFrame, args: argparse.Namespace, path: Path) -> None:
    summary = summarize(result)
    best = (
        summary[summary["model"] != "global"]
        .sort_values(["calib_n", "mae"])
        .groupby("calib_n", as_index=False)
        .first()
    )
    lines = [
        "# Steel Residual Local Calibration",
        "",
        f"- prediction_path: `{args.prediction_path}`",
        f"- dataset_path: `{args.dataset_path}`",
        f"- input_mode: `{args.input_mode}`",
        f"- target_space: `{args.target_space}`",
        f"- repeats: `{args.repeats}`",
        "",
        "## Best Mean Result By Calibration Size",
        "",
        "| calib_n | model | MAE | RMSE | R2 | MAPE | MAE improve |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in best.to_dict("records"):
        lines.append(
            f"| {int(row['calib_n'])} | {row['model']} | {row['mae']:.3f} | "
            f"{row['rmse']:.3f} | {row['r2']:.4f} | {row['mape']:.4f} | "
            f"{row['mae_improve_pct']:.1%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The `global` rows evaluate the original global steel model on the same holdout rows used by each local calibration split.",
        "Each calibrated model learns `y_true - y_global_pred` for one fixed layout only, using the selected calibration samples.",
        "A positive improvement means the local residual model reduced error relative to the original global model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


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
