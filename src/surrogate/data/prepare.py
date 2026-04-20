from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.shearwall_modeling.core.constants import MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY

PARAM_COLS = [
    "N",
    "tw_bot",
    "tw_mid",
    "tw_top",
    "hb_main",
    "bb_main",
    "hb_sec",
    "bb_sec",
    "conc_bot",
    "intensity",
    "site_class",
    "seismic_group",
]

REQUIRED_COLS = [
    "layout_id",
    "sample_id",
    "converged",
    "feasible",
    "design_passed",
    "max_drift_ratio",
    "torsion_ratio",
    "min_shear_weight_ratio",
    "min_stiffness_ratio",
    "material_steel_kg",
    "material_concrete_kg",
    "T1",
] + PARAM_COLS


@dataclass(frozen=True)
class SurrogateDataPrepConfig:
    source_dir: str = r"data\parametric\parametric_dataset_5000x143"
    output_dir: str = r"data\parametric\surrogate_dataset"
    file_glob: str = "*.parquet"
    drop_non_converged: bool = False
    deduplicate: bool = True
    drift_limit: float = 1.0 / 1000.0
    torsion_limit: float = 1.5
    stiffness_limit: float = 0.7
    default_shear_limit: float = 0.016


def prepare_training_dataset(cfg: SurrogateDataPrepConfig) -> dict[str, int | float | str]:
    source_dir = Path(cfg.source_dir)
    files = sorted(source_dir.glob(cfg.file_glob))
    if not files:
        raise ValueError(f"No parquet files found in {source_dir}.")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_parquet(file_path)
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            raise ValueError(f"{file_path.name} missing columns: {missing}")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True, copy=False)
    raw_rows = int(len(data))

    data = _normalize_columns(data)
    data = _add_labels(data, cfg)

    if cfg.drop_non_converged:
        data = data.loc[data["converged"]].copy()

    data, removed_invalid_rows = _drop_invalid_rows(data)

    removed_duplicate_rows = 0
    if cfg.deduplicate:
        before = len(data)
        data = data.drop_duplicates(subset=["layout_id"] + PARAM_COLS, keep="first")
        removed_duplicate_rows = int(before - len(data))

    data = data.sort_values(["layout_id", "sample_id"], kind="mergesort").reset_index(drop=True)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = out_dir / "surrogate_samples.parquet"
    data.to_parquet(dataset_path, index=False)

    layout_stats = _build_layout_stats(data)
    layout_stats_path = out_dir / "surrogate_layout_stats.parquet"
    layout_stats.to_parquet(layout_stats_path, index=False)

    summary = {
        "source_dir": str(source_dir),
        "output_dataset": str(dataset_path),
        "output_layout_stats": str(layout_stats_path),
        "num_source_files": len(files),
        "raw_rows": raw_rows,
        "final_rows": int(len(data)),
        "removed_invalid_rows": removed_invalid_rows,
        "removed_duplicate_rows": removed_duplicate_rows,
        "num_layouts": int(data["layout_id"].nunique()),
        "final_pass_ratio": float(data["final_pass"].mean()),
        "feasible_ratio": float(data["feasible"].mean()),
        "design_passed_ratio": float(data["design_passed"].mean()),
        "config": asdict(cfg),
    }

    summary_path = out_dir / "surrogate_data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site_class"] = out["site_class"].astype(str).str.strip().str.upper()
    out["conc_bot"] = out["conc_bot"].astype(str).str.strip().str.upper()
    for col in ("converged", "feasible", "design_passed"):
        out[col] = out[col].astype(bool)
    return out


def _add_labels(df: pd.DataFrame, cfg: SurrogateDataPrepConfig) -> pd.DataFrame:
    out = df.copy()
    out["final_pass"] = out["converged"] & out["feasible"] & out["design_passed"]
    out["margin_drift"] = cfg.drift_limit - out["max_drift_ratio"]
    out["margin_torsion"] = cfg.torsion_limit - out["torsion_ratio"]

    intensity_key = out["intensity"].astype(float).round(2)
    out["shear_weight_ratio_limit"] = intensity_key.map(MIN_SHEAR_WEIGHT_RATIO_BY_INTENSITY).fillna(
        cfg.default_shear_limit
    )
    out["margin_shear"] = out["min_shear_weight_ratio"] - out["shear_weight_ratio_limit"]

    out["stiffness_ratio_limit"] = cfg.stiffness_limit
    out["margin_stiffness"] = out["min_stiffness_ratio"] - cfg.stiffness_limit
    return out


def _drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    critical_numeric_cols = [
        "max_drift_ratio",
        "torsion_ratio",
        "min_shear_weight_ratio",
        "min_stiffness_ratio",
        "material_steel_kg",
        "material_concrete_kg",
        "T1",
    ]
    invalid = out[critical_numeric_cols].isna().any(axis=1)
    invalid |= out["material_steel_kg"] < 0.0
    invalid |= out["material_concrete_kg"] < 0.0
    invalid |= out["T1"] <= 0.0

    removed = int(invalid.sum())
    cleaned = out.loc[~invalid].copy()
    return cleaned, removed


def _build_layout_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("layout_id", as_index=False)
    layout_stats = grouped.agg(
        num_samples=("sample_id", "count"),
        final_pass_rate=("final_pass", "mean"),
        feasible_rate=("feasible", "mean"),
        design_passed_rate=("design_passed", "mean"),
        mean_steel_kg=("material_steel_kg", "mean"),
        mean_concrete_kg=("material_concrete_kg", "mean"),
        mean_max_drift=("max_drift_ratio", "mean"),
        mean_torsion=("torsion_ratio", "mean"),
    )
    return layout_stats.sort_values("layout_id").reset_index(drop=True)
