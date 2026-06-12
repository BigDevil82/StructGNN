from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOT_UTILS = ROOT / "docs" / "paper2" / "plot"
if str(PLOT_UTILS) not in sys.path:
    sys.path.insert(0, str(PLOT_UTILS))

from paper_plot_style import save_figure, set_paper_style

FEAS_BASELINE = "global_local_threshold"
STEEL_BASELINE = "global"
COLORS = {
    "Global": "#8fb6d6",
    "Best local": "#e8a69d",
    "platt": "#8fb6d6",
    "isotonic": "#efbd75",
    "logistic": "#b7d7c2",
    "residual_lgbm": "#e8a69d",
    "ridge": "#8fb6d6",
    "gpr": "#efbd75",
    "lgbm": "#b7d7c2",
    "knn": "#e8a69d",
}


def main() -> None:
    args = build_parser().parse_args()
    set_paper_style()
    root = Path(args.result_root)
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    feas = pd.read_csv(root / "feasibility" / "results.csv")
    steel = pd.read_csv(root / "steel" / "results.csv")
    feas, excluded_feas = filter_feasibility_rows(feas)

    sample = build_sample_efficiency(feas, steel)
    layout = build_layout_improvement(feas, steel, args.calib_n)
    summary = build_summary_table(feas, steel, args.calib_n)

    sample.to_csv(table_dir / "local_calibration_sample_efficiency.csv", index=False)
    excluded_feas.to_csv(table_dir / "local_calibration_excluded_feasibility_layouts.csv", index=False)
    layout["feasibility"].to_csv(
        table_dir / "local_calibration_feasibility_layout_improvement.csv", index=False
    )
    layout["steel"].to_csv(table_dir / "local_calibration_steel_layout_improvement.csv", index=False)
    summary.to_csv(table_dir / "local_calibration_summary.csv", index=False)

    plot_sample_efficiency(sample, out_dir)
    plot_layout_improvement(layout, out_dir, args.calib_n)
    print(f"[paper2][local-calib-plot] outputs written to {out_dir}")
    if not excluded_feas.empty:
        layouts = ", ".join(excluded_feas["layout_id"].astype(str).tolist())
        print(f"[paper2][local-calib-plot] excluded zero-positive feasibility layouts: {layouts}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot paper Section 3.3 local calibration figures.")
    p.add_argument("--result-root", default=r"outputs\result\paper2\local_calibration")
    p.add_argument("--out-dir", default=r"outputs\result\paper2\local_calibration\plots")
    p.add_argument("--calib-n", type=int, default=100)
    return p


def filter_feasibility_rows(feas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "positive_rate" not in feas.columns:
        return feas, pd.DataFrame(columns=["layout_id", "max_positive_rate", "rows"])

    stats = (
        feas.groupby("layout_id", as_index=False)
        .agg(max_positive_rate=("positive_rate", "max"), rows=("positive_rate", "size"))
        .sort_values("layout_id")
    )
    excluded = stats[stats["max_positive_rate"] <= 0.0].copy()
    if excluded.empty:
        return feas, excluded
    keep_layouts = set(stats.loc[stats["max_positive_rate"] > 0.0, "layout_id"].astype(str))
    filtered = feas[feas["layout_id"].astype(str).isin(keep_layouts)].copy()
    return filtered, excluded


def build_sample_efficiency(feas: pd.DataFrame, steel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feas_base = mean_by_size(feas[feas["method"] == FEAS_BASELINE], ["brier", "ece", "screen_reject_rate"])
    feas_best = best_by_size(feas[~feas["method"].str.startswith("global")], "method", "brier")
    for _, row in feas_base.iterrows():
        rows.append(
            sample_row(
                "feasibility",
                "Global",
                row,
                {"brier": "brier", "ece": "ece", "screen_reject_rate": "screen_reject_rate"},
            )
        )
    for _, row in feas_best.iterrows():
        rows.append(
            sample_row(
                "feasibility",
                "Best local",
                row,
                {"brier": "brier", "ece": "ece", "screen_reject_rate": "screen_reject_rate"},
                calibrator=str(row["method"]),
            )
        )

    steel_metrics = existing_cols(steel, ["mae", "bias", "rmse", "pair_acc", "spearman", "top_recall", "top_regret_kg"])
    steel_base = mean_by_size(steel[steel["model"] == STEEL_BASELINE], steel_metrics)
    steel_best = best_by_size(steel[steel["model"] != STEEL_BASELINE], "model", "mae")
    steel_mapping = {col: col for col in steel_metrics}
    for _, row in steel_base.iterrows():
        rows.append(sample_row("steel", "Global", row, steel_mapping))
    for _, row in steel_best.iterrows():
        rows.append(
            sample_row(
                "steel",
                "Best local",
                row,
                steel_mapping,
                calibrator=str(row["model"]),
            )
        )
    return pd.DataFrame(rows)


def mean_by_size(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("calib_n", as_index=False)[cols].mean().sort_values("calib_n")


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [col for col in cols if col in df.columns]


def best_by_size(df: pd.DataFrame, method_col: str, metric: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in {"layout_id", "repeat"}]
    mean = df.groupby(["calib_n", method_col], as_index=False)[cols].mean(numeric_only=True)
    idx = mean.groupby("calib_n")[metric].idxmin()
    return mean.loc[idx].sort_values("calib_n").reset_index(drop=True)


def sample_row(
    task: str,
    series: str,
    row: pd.Series,
    mapping: dict[str, str],
    *,
    calibrator: str = "",
) -> dict[str, float | str | int]:
    out: dict[str, float | str | int] = {
        "task": task,
        "series": series,
        "calib_n": int(row["calib_n"]),
        "calibrator": calibrator,
    }
    for out_col, src_col in mapping.items():
        out[out_col] = float(row[src_col])
    return out


def build_layout_improvement(
    feas: pd.DataFrame, steel: pd.DataFrame, calib_n: int
) -> dict[str, pd.DataFrame]:
    feas_sub = nearest_calib(feas, calib_n)
    steel_sub = nearest_calib(steel, calib_n)

    feas_best = best_by_layout(feas_sub[~feas_sub["method"].str.startswith("global")], "method", "brier")
    feas_base = layout_mean(
        feas_sub[feas_sub["method"] == FEAS_BASELINE], ["brier", "ece", "screen_reject_rate"]
    )
    feas_layout = feas_best.merge(feas_base, on="layout_id", suffixes=("_calib", "_global"))
    feas_layout["brier_improve_pct"] = (
        feas_layout["brier_global"] - feas_layout["brier_calib"]
    ) / feas_layout["brier_global"].clip(lower=1.0e-9)

    steel_best = best_by_layout(steel_sub[steel_sub["model"] != STEEL_BASELINE], "model", "mae")
    steel_base = layout_mean(steel_sub[steel_sub["model"] == STEEL_BASELINE], ["mae", "rmse", "bias"])
    steel_layout = steel_best.merge(steel_base, on="layout_id", suffixes=("_calib", "_global"))
    steel_layout["mae_improve_pct"] = (steel_layout["mae_global"] - steel_layout["mae_calib"]) / steel_layout[
        "mae_global"
    ].clip(lower=1.0e-9)
    return {"feasibility": feas_layout, "steel": steel_layout}


def nearest_calib(df: pd.DataFrame, calib_n: int) -> pd.DataFrame:
    sizes = np.asarray(sorted(df["calib_n"].unique()), dtype=int)
    nearest = int(sizes[np.argmin(np.abs(sizes - calib_n))])
    return df[df["calib_n"] == nearest].copy()


def best_by_layout(df: pd.DataFrame, method_col: str, metric: str) -> pd.DataFrame:
    mean = df.groupby(["layout_id", method_col], as_index=False).mean(numeric_only=True)
    idx = mean.groupby("layout_id")[metric].idxmin()
    return mean.loc[idx].reset_index(drop=True)


def layout_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.groupby("layout_id", as_index=False)[cols].mean()


def build_summary_table(feas: pd.DataFrame, steel: pd.DataFrame, calib_n: int) -> pd.DataFrame:
    rows = []
    fsub = nearest_calib(feas, calib_n)
    ssub = nearest_calib(steel, calib_n)
    fbase = fsub[fsub["method"] == FEAS_BASELINE]
    fbest = best_by_size(fsub[~fsub["method"].str.startswith("global")], "method", "brier").iloc[0]
    sbase = ssub[ssub["model"] == STEEL_BASELINE]
    sbest = best_by_size(ssub[ssub["model"] != STEEL_BASELINE], "model", "mae").iloc[0]
    rows.append(
        {
            "task": "Feasibility",
            "metric": "Brier",
            "global": fbase["brier"].mean(),
            "calibrated": fbest["brier"],
            "improvement_pct": (fbase["brier"].mean() - fbest["brier"]) / max(fbase["brier"].mean(), 1.0e-9),
            "calib_n": int(fbest["calib_n"]),
            "calibrator": fbest["method"],
        }
    )
    rows.append(
        {
            "task": "Feasibility",
            "metric": "ECE",
            "global": fbase["ece"].mean(),
            "calibrated": fbest["ece"],
            "improvement_pct": (fbase["ece"].mean() - fbest["ece"]) / max(fbase["ece"].mean(), 1.0e-9),
            "calib_n": int(fbest["calib_n"]),
            "calibrator": fbest["method"],
        }
    )
    for metric, label, higher_is_better in [
        ("recall", "Recall", True),
        ("pr_auc", "PR-AUC", True),
        ("screen_reject_rate", "Screen reject rate", True),
        ("screen_recall", "Screen recall", True),
    ]:
        rows.append(
            {
                "task": "Feasibility",
                "metric": label,
                "global": fbase[metric].mean(),
                "calibrated": fbest[metric],
                "improvement_pct": relative_change(fbase[metric].mean(), fbest[metric], higher_is_better),
                "calib_n": int(fbest["calib_n"]),
                "calibrator": fbest["method"],
            }
        )
    rows.append(
        {
            "task": "Steel",
            "metric": "MAE",
            "global": sbase["mae"].mean(),
            "calibrated": sbest["mae"],
            "improvement_pct": (sbase["mae"].mean() - sbest["mae"]) / max(sbase["mae"].mean(), 1.0e-9),
            "calib_n": int(sbest["calib_n"]),
            "calibrator": sbest["model"],
        }
    )
    for metric, label, higher_is_better in [
        ("rmse", "RMSE", False),
        ("r2", "R2", True),
        ("mape", "MAPE", False),
    ]:
        rows.append(
            {
                "task": "Steel",
                "metric": label,
                "global": sbase[metric].mean(),
                "calibrated": sbest[metric],
                "improvement_pct": relative_change(sbase[metric].mean(), sbest[metric], higher_is_better),
                "calib_n": int(sbest["calib_n"]),
                "calibrator": sbest["model"],
            }
        )
    for metric, label, higher_is_better in [
        ("pair_acc", "Pairwise accuracy", True),
        ("spearman", "Spearman", True),
        ("top_recall", "Top-10% recall", True),
        ("top_regret_kg", "Top-10% regret", False),
    ]:
        if metric not in sbase.columns or metric not in sbest.index:
            continue
        rows.append(
            {
                "task": "Steel",
                "metric": label,
                "global": sbase[metric].mean(),
                "calibrated": sbest[metric],
                "improvement_pct": relative_change(sbase[metric].mean(), sbest[metric], higher_is_better),
                "calib_n": int(sbest["calib_n"]),
                "calibrator": sbest["model"],
            }
        )
    rows.append(
        {
            "task": "Steel",
            "metric": "Bias",
            "global": sbase["bias"].mean(),
            "calibrated": sbest["bias"],
            "improvement_pct": (abs(sbase["bias"].mean()) - abs(sbest["bias"]))
            / max(abs(sbase["bias"].mean()), 1.0e-9),
            "calib_n": int(sbest["calib_n"]),
            "calibrator": sbest["model"],
        }
    )
    return pd.DataFrame(rows)


def relative_change(global_value: float, calibrated_value: float, higher_is_better: bool) -> float:
    denom = max(abs(float(global_value)), 1.0e-9)
    if higher_is_better:
        return (float(calibrated_value) - float(global_value)) / denom
    return (float(global_value) - float(calibrated_value)) / denom


def plot_sample_efficiency(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))
    plot_metric_lines(axes[0, 0], df[df["task"] == "feasibility"], "brier", "Brier score")
    plot_metric_lines(
        axes[0, 1],
        df[df["task"] == "feasibility"],
        "screen_reject_rate",
        "Reject rate (%)",
        value_scale=100.0,
    )
    plot_metric_lines(axes[1, 0], df[df["task"] == "steel"], "mae", "MAE (t)", value_scale=1.0 / 1000.0)
    plot_metric_lines(
        axes[1, 1],
        df[df["task"] == "steel"],
        "bias",
        "Bias (t)",
        value_scale=1.0 / 1000.0,
        zero_line=True,
    )
    captions = [
        "(a) Feasibility probability calibration error",
        "(b) Screening capacity at high feasible recall",
        "(c) Steel residual prediction error",
        "(d) Steel residual prediction bias",
    ]
    for ax, caption in zip(axes.ravel(), captions):
        add_caption(ax, caption)
    fig.suptitle("Sample Efficiency of Layout-Local Calibration", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.91, wspace=0.28, hspace=0.35)
    save_figure(fig, "main_3_3_local_calibration_sample_efficiency.png", out_dir)


def plot_metric_lines(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    value_scale: float = 1.0,
    zero_line: bool = False,
) -> None:
    if metric not in df.columns:
        ax.text(0.5, 0.5, f"{metric} not available", transform=ax.transAxes, ha="center", va="center")
        ax.set_xlabel("Local FEA samples")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        return
    for series, part in df.groupby("series", sort=False):
        color = COLORS.get(series, "#999999")
        ax.plot(
            part["calib_n"],
            part[metric] * value_scale,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=color,
            label=series,
        )
    if zero_line:
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Local FEA samples")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend(frameon=False)


def plot_layout_improvement(layout: dict[str, pd.DataFrame], out_dir: Path, calib_n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    plot_before_after(
        axes[0],
        layout["feasibility"],
        "brier_global",
        "brier_calib",
        "Global Brier",
        "Calibrated Brier",
    )
    plot_before_after(
        axes[1],
        layout["steel"],
        "mae_global",
        "mae_calib",
        "Global MAE (t)",
        "Calibrated MAE (t)",
        value_scale=1.0 / 1000.0,
    )
    add_caption(axes[0], "(a) Layout-wise feasibility calibration improvement")
    add_caption(axes[1], "(b) Layout-wise steel residual calibration improvement")
    fig.suptitle(f"Layout-Wise Improvement with {calib_n} Local FEA Samples", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.88, wspace=0.28)
    save_figure(fig, "main_3_3_local_calibration_layout_improvement.png", out_dir)


def plot_before_after(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    *,
    value_scale: float = 1.0,
) -> None:
    x = df[x_col].to_numpy(dtype=float) * value_scale
    y = df[y_col].to_numpy(dtype=float) * value_scale
    ax.scatter(x, y, s=34, color="#e8a69d", edgecolor="#333333", linewidth=0.4, alpha=0.8)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = 0.04 * max(hi - lo, 1.0e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#777777", linewidth=1.0, linestyle="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def add_caption(ax: plt.Axes, text: str) -> None:
    ax.text(0.5, -0.22, text, transform=ax.transAxes, ha="center", va="top", fontsize=9)


if __name__ == "__main__":
    main()
