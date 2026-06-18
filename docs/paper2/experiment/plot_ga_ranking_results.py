from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PLOT_UTILS = PROJECT_ROOT / "docs" / "paper2" / "plot"
if str(PLOT_UTILS) not in sys.path:
    sys.path.insert(0, str(PLOT_UTILS))

from paper_plot_style import set_paper_style

ROOT = Path(r"outputs\result\optimization\ranking_overnight")
FIG_DIR = ROOT / "figures"
SMALL_LAYOUTS = ["L17_101", "L17_123", "L17_145", "L17_208"]
SMALL_SEEDS = [42, 7]
METHOD_LABELS = {
    "full": "Full GA-FEA",
    "gnn_cost": "Surrogate cost",
    "gnn_screen_cost": "Screen + cost",
}
METHOD_COLORS = {
    "full": "#4C78A8",
    "gnn_cost": "#F58518",
    "gnn_screen_cost": "#B279A2",
}


def main() -> None:
    set_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_primary_summary()
    _plot_primary_paired_distributions()
    _plot_eval_ratio_tradeoff()
    _plot_layout_level_tradeoff()
    _plot_combined_figure()
    print(f"figures saved to {FIG_DIR}")


def _plot_primary_summary() -> None:
    df = pd.read_csv(ROOT / "primary_large" / "summary_by_method.csv")
    df["label"] = df["method"].map(METHOD_LABELS)
    colors = [METHOD_COLORS.get(m, "#777777") for m in df["method"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].bar(df["label"], df["mean_fea_calls"], color=colors)
    axes[0].set_ylabel("Mean real FEA calls")
    axes[0].set_title("Computational cost")

    axes[1].bar(df["label"], df["feasible_rate"] * 100, color=colors)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Feasible rate (%)")
    axes[1].set_title("Optimization success")

    axes[2].bar(df["label"], df["mean_best_objective"] / 1000.0, color=colors)
    axes[2].set_ylabel("Mean best objective (x1000)")
    axes[2].set_title("Objective quality")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "primary_summary.png", dpi=220)
    plt.close(fig)


def _plot_primary_paired_distributions() -> None:
    df = pd.read_csv(ROOT / "primary_large" / "paired_summary.csv")
    methods = [m for m in ["gnn_cost", "gnn_screen_cost"] if m in set(df["method"])]
    labels = [METHOD_LABELS[m] for m in methods]
    both = df[df["full_feasible"] & df["method_feasible"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    objective_data = [both[both["method"] == m]["objective_ratio"].to_numpy() for m in methods]
    reduction_data = [df[df["method"] == m]["fea_reduction"].to_numpy() * 100.0 for m in methods]

    axes[0].boxplot(objective_data, tick_labels=labels, showmeans=True)
    axes[0].axhline(1.0, color="#888888", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Objective ratio vs full GA-FEA")
    axes[0].set_title("Quality among both-feasible pairs")

    axes[1].boxplot(reduction_data, tick_labels=labels, showmeans=True)
    axes[1].set_ylabel("FEA reduction (%)")
    axes[1].set_title("FEA savings")

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "primary_paired_distributions.png", dpi=220)
    plt.close(fig)


def _plot_eval_ratio_tradeoff() -> None:
    rows = []
    for ratio, exp in [(0.25, "ratio_025"), (0.75, "ratio_075")]:
        rows.extend(_ratio_rows(ROOT / exp / "paired_summary.csv", ratio))
    primary = pd.read_csv(ROOT / "primary_large" / "paired_summary.csv")
    primary = primary[primary["layout_id"].isin(SMALL_LAYOUTS) & primary["seed"].isin(SMALL_SEEDS)]
    rows.extend(_ratio_rows(primary, 0.5))
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for method in [m for m in ["gnn_cost", "gnn_screen_cost"] if m in set(df["method"])]:
        g = df[df["method"] == method].sort_values("eval_ratio")
        label = METHOD_LABELS[method]
        color = METHOD_COLORS[method]
        axes[0].plot(g["eval_ratio"], g["fea_reduction"] * 100, marker="o", label=label, color=color)
        axes[1].plot(g["eval_ratio"], g["feasible_rate"] * 100, marker="o", label=label, color=color)
        axes[2].plot(g["eval_ratio"], g["both_feasible_median_ratio"], marker="o", label=label, color=color)

    axes[0].set_ylabel("FEA reduction (%)")
    axes[0].set_title("Cost saving")
    axes[1].set_ylabel("Feasible rate (%)")
    axes[1].set_title("Success rate")
    axes[2].axhline(1.0, color="#888888", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Median objective ratio")
    axes[2].set_title("Both-feasible quality")
    for ax in axes:
        ax.set_xlabel("Evaluated population ratio")
        ax.set_xticks([0.25, 0.5, 0.75])
        ax.grid(alpha=0.25)
    axes[2].legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_ratio_tradeoff.png", dpi=220)
    plt.close(fig)


def _ratio_rows(data: Path | pd.DataFrame, ratio: float) -> list[dict[str, float | str]]:
    df = pd.read_csv(data) if isinstance(data, Path) else data
    rows = []
    for method, g in df.groupby("method"):
        both = g[g["full_feasible"] & g["method_feasible"]]
        rows.append(
            {
                "eval_ratio": ratio,
                "method": method,
                "fea_reduction": float(g["fea_reduction"].mean()),
                "feasible_rate": float(g["method_feasible"].mean()),
                "both_feasible_median_ratio": float(both["objective_ratio"].median()) if len(both) else float("nan"),
            }
        )
    return rows


def _plot_layout_level_tradeoff() -> None:
    df = pd.read_csv(ROOT / "primary_large" / "paired_summary.csv")
    gnn = df[df["method"] == "gnn_screen_cost"].copy()
    if gnn.empty:
        gnn = df[df["method"] == "gnn_cost"].copy()
    agg = (
        gnn.groupby("layout_id")
        .agg(
            feasible_rate=("method_feasible", "mean"),
            full_feasible_rate=("full_feasible", "mean"),
            mean_fea_reduction=("fea_reduction", "mean"),
            median_objective_ratio=("objective_ratio", "median"),
        )
        .reset_index()
        .sort_values("median_objective_ratio")
    )

    fig, ax1 = plt.subplots(figsize=(10, 4.2))
    x = range(len(agg))
    ax1.bar(x, agg["mean_fea_reduction"] * 100.0, color=METHOD_COLORS["gnn_screen_cost"], alpha=0.85)
    ax1.set_ylabel("Mean FEA reduction (%)")
    ax1.set_ylim(0, max(55, agg["mean_fea_reduction"].max() * 120))
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(agg["layout_id"], rotation=35, ha="right")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(list(x), agg["median_objective_ratio"], color="#333333", marker="o", linewidth=1.5)
    ax2.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax2.set_ylabel("Median objective ratio")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "layout_level_gnn_tradeoff.png", dpi=220)
    plt.close(fig)


def _plot_combined_figure() -> None:
    items = [
        ("Overall performance", FIG_DIR / "primary_summary.png"),
        ("Paired comparison", FIG_DIR / "primary_paired_distributions.png"),
        ("Evaluation-ratio tradeoff", FIG_DIR / "eval_ratio_tradeoff.png"),
        ("Layout-level tradeoff", FIG_DIR / "layout_level_gnn_tradeoff.png"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    for ax, (title, path) in zip(axes.ravel(), items):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title, fontsize=13, pad=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_optimization_results.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
