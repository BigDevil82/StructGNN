from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from paper_plot_style import (
    ALGORITHM_ORDER,
    METHOD_ORDER,
    ROOT,
    SURROGATE_METHODS,
    family_of,
)


DEFAULT_EXPERIMENTS = {
    "GA": ROOT / "outputs/result/optimization/ga_primary_pop48_gen15_parallel",
    "PSO": ROOT / "outputs/result/optimization/pso_primary",
    "Random Search": ROOT / "outputs/result/optimization/random_primary_t720_b48",
}
SELECTED15_EXPERIMENTS = {
    "GA": ROOT / "outputs/result/optimization/ga_primary_selected15",
    "PSO": ROOT / "outputs/result/optimization/pso_primary_selected15",
    "Random Search": ROOT / "outputs/result/optimization/random_primary_selected15",
}
SENTINEL_OBJECTIVE = 1.0e10


def load_all_summaries(experiments: dict[str, Path] | None = None) -> pd.DataFrame:
    experiments = experiments or DEFAULT_EXPERIMENTS
    frames: list[pd.DataFrame] = []
    for algorithm, exp_dir in experiments.items():
        frame = load_summary(exp_dir)
        frame["algorithm"] = algorithm
        frame["experiment_dir"] = str(exp_dir)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_summary(exp_dir: str | Path) -> pd.DataFrame:
    exp_dir = Path(exp_dir)
    summary_path = exp_dir / "primary_large" / "summary.csv"
    if not summary_path.exists():
        matches = sorted(exp_dir.rglob("summary.csv"))
        if not matches:
            raise FileNotFoundError(f"summary.csv not found under {exp_dir}")
        summary_path = matches[0]

    df = pd.read_csv(summary_path)
    if "ok" in df.columns:
        df = df[df["ok"].astype(str).str.lower().isin(["true", "1"])].copy()
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    for col in [
        "seed",
        "best_objective",
        "material_cost",
        "fea_calls",
        "first_feasible_fea_calls",
        "screened_ratio",
        "cost_preselect_skipped_ratio",
        "local_calibration_samples",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["best_feasible"] = df["best_feasible"].astype(str).str.lower().isin(["true", "1"])
    df["layout_id"] = df["layout_id"].astype(str)
    df["family"] = df["layout_id"].map(family_of)
    return df


def paired_with_full(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["algorithm", "layout_id", "seed"]
    full_cols = keys + ["best_feasible", "material_cost", "best_objective", "fea_calls", "first_feasible_fea_calls"]
    full = df[df["method"] == "full"][full_cols].rename(
        columns={
            "best_feasible": "full_feasible",
            "material_cost": "full_material_cost",
            "best_objective": "full_objective",
            "fea_calls": "full_fea_calls",
            "first_feasible_fea_calls": "full_first_feasible_fea_calls",
        }
    )
    other = df[df["method"].isin(SURROGATE_METHODS)].copy()
    paired = other.merge(full, on=keys, how="inner")
    paired["case_id"] = paired["layout_id"] + "_seed" + paired["seed"].astype(int).astype(str)
    paired["fea_ratio"] = paired["fea_calls"] / paired["full_fea_calls"]
    paired["fea_reduction"] = 1.0 - paired["fea_ratio"]
    paired["both_feasible"] = paired["best_feasible"] & paired["full_feasible"]
    paired["cost_ratio"] = np.where(
        paired["both_feasible"],
        paired["material_cost"] / paired["full_material_cost"],
        np.nan,
    )
    paired["cost_gap_pct"] = 100.0 * (paired["cost_ratio"] - 1.0)
    return paired


def find_result_json(row: pd.Series) -> Path | None:
    exp_dir = Path(str(row["experiment_dir"]))
    root = exp_dir / "primary_large"
    name = f"{row['layout_id']}_seed{int(row['seed'])}_{row['method']}.json"
    path = root / name
    if path.exists():
        return path

    result_path = row.get("result_path")
    if isinstance(result_path, str) and result_path:
        p = ROOT / result_path
        if p.exists():
            return p
    matches = sorted(root.glob(f"{row['layout_id']}_seed{int(row['seed'])}_{row['method']}.json"))
    return matches[0] if matches else None


def read_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def material_points(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_i = 0
    best_feasible = np.inf
    for step_i, step in enumerate(history, start=1):
        population = step.get("population")
        if not isinstance(population, list):
            continue
        for item in population:
            status = candidate_status(item)
            if status in {"screened", "cost_skip"}:
                continue
            value = safe_float(item.get("material_cost"))
            if not np.isfinite(value) or abs(value) >= SENTINEL_OBJECTIVE:
                continue
            eval_i += 1
            feasible = bool(item.get("feasible", False))
            if feasible:
                best_feasible = min(best_feasible, value)
            rows.append(
                {
                    "evaluation": eval_i,
                    "step": step_i,
                    "material_cost": value,
                    "feasible": feasible,
                    "best_feasible_cost": best_feasible if np.isfinite(best_feasible) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def screening_flow(history: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for step_i, step in enumerate(history, start=1):
        population = step.get("population")
        if not isinstance(population, list) or not population:
            continue
        counts = {"screened": 0, "cost_skip": 0, "fea": 0, "feasible": 0}
        for item in population:
            status = candidate_status(item)
            counts[status] += 1
            if status == "fea" and bool(item.get("feasible", False)):
                counts["feasible"] += 1
        total = counts["screened"] + counts["cost_skip"] + counts["fea"]
        if total <= 0:
            continue
        rows.append(
            {
                "step": step_i,
                "screened": counts["screened"] / total,
                "cost_skip": counts["cost_skip"] / total,
                "fea": counts["fea"] / total,
                "feasible": counts["feasible"] / total,
                "n": total,
            }
        )
    return pd.DataFrame(rows)


def candidate_status(item: dict[str, Any]) -> str:
    if bool(item.get("surrogate_screen_reject", False)):
        return "screened"
    if bool(item.get("cost_preselect_skip", False)):
        return "cost_skip"
    return "fea"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def complete_case_rows(df: pd.DataFrame, algorithm: str = "GA") -> pd.DataFrame:
    sub = df[df["algorithm"] == algorithm].copy()
    counts = sub.groupby(["layout_id", "seed"])["method"].nunique()
    complete = counts[counts >= len(METHOD_ORDER)].reset_index()[["layout_id", "seed"]]
    return sub.merge(complete, on=["layout_id", "seed"], how="inner")


def sort_layouts_for_heatmap(df: pd.DataFrame) -> list[str]:
    stats = (
        df[df["method"] == "full"]
        .groupby(["family", "layout_id"])["best_feasible"]
        .mean()
        .reset_index(name="full_success")
    )
    stats["family_order"] = stats["family"].map({"L17": 0, "L27": 1, "L1L28": 2}).fillna(9)
    stats["layout_num"] = stats["layout_id"].str.extract(r"_(\d+)").astype(float)
    stats = stats.sort_values(["family_order", "full_success", "layout_num"], ascending=[True, False, True])
    return stats["layout_id"].tolist()


def algorithm_ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(ALGORITHM_ORDER)}
    return df.assign(_algo_order=df["algorithm"].map(order).fillna(99)).sort_values("_algo_order")
