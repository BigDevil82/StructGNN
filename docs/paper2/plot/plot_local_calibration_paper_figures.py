from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgba

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOT_UTILS = ROOT / "docs" / "paper2" / "plot"
if str(PLOT_UTILS) not in sys.path:
    sys.path.insert(0, str(PLOT_UTILS))

from paper_plot_style import PANEL_CAPTION_SIZE, PLOT_DIR, save_figure, set_paper_style

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
    p.add_argument("--out-dir", default=str(PLOT_DIR))
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

    steel_metrics = existing_cols(
        steel, ["mae", "bias", "rmse", "pair_acc", "spearman", "top_recall", "top_regret_kg"]
    )
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
    feasibility = df[df["task"] == "feasibility"].copy()
    plot_feasibility_lollipop(
        axes[0, 0],
        feasibility,
        "brier",
        "Brier reduction (%)",
        higher_is_better=False,
    )
    plot_feasibility_lollipop(
        axes[0, 1],
        feasibility,
        "screen_reject_rate",
        "Reject-rate gain (pp)",
        higher_is_better=True,
        value_scale=100.0,
        unit=" pp",
    )
    steel = df[df["task"] == "steel"].copy()
    plot_steel_mae_reduction(axes[1, 0], steel)
    plot_steel_local_bias(axes[1, 1], steel)
    captions = [
        "(a) Feasibility probability calibration improvement",
        "(b) Screening capacity improvement at high recall",
        "(c) Steel MAE reduction relative to global baseline",
        "(d) Steel bias after local calibration",
    ]
    for ax, caption in zip(axes.ravel(), captions):
        add_caption(ax, caption)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.13, top=0.98, wspace=0.20, hspace=0.42)
    save_figure(fig, "calibration_sample_efficiency.png", out_dir)


def plot_feasibility_lollipop(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    *,
    higher_is_better: bool,
    value_scale: float = 1.0,
    unit: str = "%",
) -> None:
    pivot = df.pivot(index="calib_n", columns="series", values=metric).sort_index()
    if "Global" not in pivot.columns or "Best local" not in pivot.columns:
        ax.text(0.5, 0.5, f"{metric} not available", transform=ax.transAxes, ha="center", va="center")
        return

    if higher_is_better:
        values = (pivot["Best local"] - pivot["Global"]) * value_scale
        zero_line = 0.0
    else:
        values = 100.0 * (pivot["Global"] - pivot["Best local"]) / pivot["Global"].clip(lower=1.0e-9)
        zero_line = 0.0

    x = np.arange(len(values))
    colors = improvement_colors(values.to_numpy(dtype=float))
    for xi, value, color in zip(x, values.to_numpy(dtype=float), colors):
        ax.vlines(xi, zero_line, value, color=color, linewidth=3.0, alpha=0.78)
    ax.scatter(
        x, values.to_numpy(dtype=float), s=54, color=colors, edgecolor="#333333", linewidth=0.5, zorder=3
    )
    ax.axhline(zero_line, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in values.index])
    ax.set_xlabel("Local FEA samples")
    ax.set_ylabel(ylabel)
    ymin = min(0.0, float(values.min()) * 1.20)
    ymax = max(0.0, float(values.max()) * 1.18)
    if ymax <= ymin:
        ymax = ymin + 1.0
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.55, len(values) - 0.45)
    add_soft_gradient(ax)
    ax.grid(axis="y")

    for xi, value in zip(x, values.to_numpy(dtype=float)):
        if value >= 0:
            va = "bottom"
            y = value + 0.03 * (ymax - ymin)
        else:
            va = "top"
            y = value - 0.03 * (ymax - ymin)
        text = f"{value:.1f}{unit}" if unit else f"{value:.1f}"
        ax.text(xi, y, text, ha="center", va=va, fontsize=8, color="#333333")


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


def add_soft_gradient(ax: plt.Axes, color: str = "#75b7d8", alpha_top: float = 0.16) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    rgba = np.array(to_rgba(color))
    gradient = np.ones((256, 1, 4), dtype=float)
    gradient[:, :, :3] = rgba[:3]
    gradient[:, :, 3] = np.linspace(0.02, alpha_top, 256)[:, None]
    ax.imshow(
        gradient,
        extent=(x0, x1, y0, y1),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def improvement_colors(values: np.ndarray) -> list[tuple[float, float, float, float]]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return [to_rgba("#75b7d8") for _ in values]
    lo = float(finite.min())
    hi = float(finite.max())
    denom = max(hi - lo, 1.0e-9)
    norm = np.clip((values - lo) / denom, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list("improvement_blue", ["#cfe8f3", "#75b7d8", "#3288bd"])
    return [cmap(float(v)) for v in norm]


def plot_steel_mae_reduction(ax: plt.Axes, df: pd.DataFrame) -> None:
    pivot = df.pivot(index="calib_n", columns="series", values="mae").sort_index()
    if "Global" not in pivot.columns or "Best local" not in pivot.columns:
        ax.text(0.5, 0.5, "Steel MAE data not available", transform=ax.transAxes, ha="center", va="center")
        return
    reduction = 100.0 * (pivot["Global"] - pivot["Best local"]) / pivot["Global"].clip(lower=1.0e-9)
    x = np.arange(len(reduction))
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in reduction.index])
    ax.set_xlabel("Local FEA samples")
    ax.set_ylabel("MAE reduction (%)")
    ax.set_ylim(0, min(100.0, max(90.0, float(reduction.max()) * 1.12)))
    ax.set_xlim(-0.55, len(reduction) - 0.45)
    add_soft_gradient(ax)
    colors = improvement_colors(reduction.to_numpy(dtype=float))
    bars = ax.bar(
        x,
        reduction.to_numpy(),
        color=colors,
        edgecolor="#ffffff",
        linewidth=0.8,
        zorder=2,
    )
    ax.grid(axis="y")

    first_n = int(reduction.index[0])
    first_global = float(pivot.loc[first_n, "Global"] / 1000.0)
    first_local = float(pivot.loc[first_n, "Best local"] / 1000.0)
    first_reduction = float(reduction.iloc[0])
    ax.text(
        x[0],
        first_reduction + 1.0,
        f"{first_reduction:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.text(
        x[0],
        first_reduction + 8.0,
        f"{first_global:.2f}->{first_local:.2f} t",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    for bar, value in zip(bars[1:], reduction.iloc[1:]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 1.0,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def plot_steel_local_bias(ax: plt.Axes, df: pd.DataFrame) -> None:
    pivot = df.pivot(index="calib_n", columns="series", values="bias").sort_index()
    if "Global" not in pivot.columns or "Best local" not in pivot.columns:
        ax.text(0.5, 0.5, "Steel bias data not available", transform=ax.transAxes, ha="center", va="center")
        return
    x = np.arange(len(pivot.index))
    local_bias = pivot["Best local"] / 1000.0
    global_bias = float(pivot["Global"].mean() / 1000.0)
    ax.plot(
        x,
        local_bias.to_numpy(),
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        color="#75b7d8",
        label="Best local",
    )
    ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in pivot.index])
    ax.set_xlabel("Local FEA samples")
    ax.set_ylabel("Bias (t)")
    max_abs = max(float(local_bias.abs().max()) * 1.45, 0.08)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_xlim(-0.55, len(pivot.index) - 0.45)
    add_soft_gradient(ax)
    ax.grid(axis="y")
    ax.text(
        0.5,
        0.93,
        f"Global bias = {global_bias:.2f} t",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    first_n = int(pivot.index[0])
    correction = (
        100.0
        * (abs(float(pivot.loc[first_n, "Global"])) - abs(float(pivot.loc[first_n, "Best local"])))
        / max(abs(float(pivot.loc[first_n, "Global"])), 1.0e-9)
    )
    ax.text(
        x[0],
        float(local_bias.iloc[0]) + 0.005,
        f"{correction:.1f}% corrected",
        ha="left",
        va="bottom" if local_bias.iloc[0] >= 0 else "top",
        fontsize=10,
        color="#333333",
    )


def plot_layout_improvement(layout: dict[str, pd.DataFrame], out_dir: Path, calib_n: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.4))
    plot_layout_dumbbell(
        axes[0],
        layout["feasibility"],
        "brier_global",
        "brier_calib",
        "Brier score",
        sort_col="brier_global",
    )
    plot_layout_dumbbell(
        axes[1],
        layout["steel"],
        "mae_global",
        "mae_calib",
        "MAE (t)",
        sort_col="mae_global",
        value_scale=1.0 / 1000.0,
        show_ylabels=False,
    )
    add_caption(axes[0], "(a) Layout-wise feasibility calibration improvement", y=-0.14)
    add_caption(axes[1], "(b) Layout-wise steel residual calibration improvement", y=-0.14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.15, top=0.91, wspace=0.10)
    save_figure(fig, "local_calibration_effect.png", out_dir)


def plot_layout_dumbbell(
    ax: plt.Axes,
    df: pd.DataFrame,
    global_col: str,
    calibrated_col: str,
    xlabel: str,
    *,
    sort_col: str,
    value_scale: float = 1.0,
    show_ylabels: bool = True,
) -> None:
    data = df.sort_values(sort_col, ascending=False).reset_index(drop=True).copy()
    y = np.arange(len(data))
    x_global = data[global_col].to_numpy(dtype=float) * value_scale
    x_calib = data[calibrated_col].to_numpy(dtype=float) * value_scale
    improved = x_calib <= x_global
    line_colors = np.where(improved, "#75b7d8", "#e8a69d")

    for yi, x0, x1, color in zip(y, x_global, x_calib, line_colors):
        ax.plot([x0, x1], [yi, yi], color=color, linewidth=2.0, alpha=0.75, zorder=1)
    ax.scatter(
        x_global,
        y,
        s=34,
        color="#d9d9d9",
        edgecolor="#555555",
        linewidth=0.4,
        label="Global",
        zorder=2,
    )
    ax.scatter(
        x_calib,
        y,
        s=38,
        color="#75b7d8",
        edgecolor="#333333",
        linewidth=0.4,
        label="Local calibrated",
        zorder=3,
    )
    labels = layout_display_labels(data["layout_id"].astype(str).tolist())
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    lo = float(min(x_global.min(), x_calib.min()))
    hi = float(max(x_global.max(), x_calib.max()))
    pad = 0.06 * max(hi - lo, 1.0e-9)
    ax.set_xlim(max(0.0, lo - pad), hi + pad)


def layout_display_labels(layout_ids: list[str]) -> list[str]:
    counters: dict[str, int] = {}
    labels: list[str] = []
    for layout_id in layout_ids:
        counters[layout_id] = counters.get(layout_id, 0) + 1
        labels.append(f"L{len(labels) + 1}")
    return labels


def add_caption(ax: plt.Axes, text: str, *, y: float = -0.25) -> None:
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=PANEL_CAPTION_SIZE)


if __name__ == "__main__":
    main()
