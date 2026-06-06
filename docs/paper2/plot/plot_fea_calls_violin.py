from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_EXPERIMENTS = {
    "GA": ROOT / "outputs/result/optimization/ga_primary_pop48_gen15_parallel",
    "PSO": ROOT / "outputs/result/optimization/pso_primary",
    "Random Search": ROOT / "outputs/result/optimization/random_primary_t720_b48",
}
METHOD_ORDER = ["full", "gnn_cost", "gnn_screen_cost"]
METHOD_LABELS = {
    "full": "Full FEA",
    "gnn_cost": "Cost\nsurrogate",
    "gnn_screen_cost": "Screen + cost\nsurrogate",
}
COLORS = ["#7f8c8d", "#4c78a8", "#f58518"]


def main() -> None:
    args = build_parser().parse_args()
    experiments = {
        "GA": Path(args.ga_dir),
        "PSO": Path(args.pso_dir),
        "Random Search": Path(args.random_dir),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    y_max = 0.0
    for ax, (name, exp_dir) in zip(axes, experiments.items()):
        df = load_summary(exp_dir)
        y_max = max(y_max, float(df["fea_calls"].max()))
        plot_algorithm(ax, df, name)

    axes[0].set_ylabel("FEA calls")
    for ax in axes:
        ax.set_ylim(0, y_max * 1.08)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("FEA Calls by Optimization Algorithm and Surrogate Strategy", y=1.02, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / args.filename
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"[plot] saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot violin distributions of FEA calls for optimization experiments.")
    p.add_argument("--ga-dir", default=str(DEFAULT_EXPERIMENTS["GA"]))
    p.add_argument("--pso-dir", default=str(DEFAULT_EXPERIMENTS["PSO"]))
    p.add_argument("--random-dir", default=str(DEFAULT_EXPERIMENTS["Random Search"]))
    p.add_argument("--out-dir", default=str(ROOT / "outputs/result/optimization/plots"))
    p.add_argument("--filename", default="fea_calls_violin.png")
    p.add_argument("--dpi", type=int, default=300)
    return p


def load_summary(exp_dir: Path) -> pd.DataFrame:
    summary_path = exp_dir / "primary_large" / "summary.csv"
    if not summary_path.exists():
        matches = sorted(exp_dir.rglob("summary.csv"))
        if not matches:
            raise FileNotFoundError(f"summary.csv not found under {exp_dir}")
        summary_path = matches[0]

    df = pd.read_csv(summary_path)
    if "ok" in df.columns:
        df = df[df["ok"].astype(str).str.lower().isin(["true", "1"])]
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    df["fea_calls"] = pd.to_numeric(df["fea_calls"], errors="coerce")
    df = df.dropna(subset=["fea_calls"])
    if df.empty:
        raise ValueError(f"No usable FEA call data in {summary_path}")
    return df


def plot_algorithm(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    data = [df.loc[df["method"] == method, "fea_calls"].to_numpy() for method in METHOD_ORDER]
    positions = range(1, len(METHOD_ORDER) + 1)
    parts = ax.violinplot(data, positions=positions, widths=0.78, showmeans=False, showmedians=False)

    for body, color in zip(parts["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
        body.set_linewidth(0.8)
    for key in ["cbars", "cmins", "cmaxes"]:
        parts[key].set_color("#555555")
        parts[key].set_linewidth(0.8)

    rng = np.random.default_rng(20260606)
    medians = [pd.Series(values).median() for values in data]
    for x, values, color, med in zip(positions, data, COLORS, medians):
        jitter = rng.uniform(-0.085, 0.085, size=len(values))
        ax.scatter(
            x + jitter,
            values,
            s=15,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.62,
            zorder=3,
        )
        ax.hlines(med, x - 0.18, x + 0.18, color="#111111", linewidth=1.3, zorder=4)
        ax.text(x, med, f"{med:.0f}", ha="center", va="bottom", fontsize=8, color="#111111")

    ax.set_title(title, fontsize=12)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([METHOD_LABELS[x] for x in METHOD_ORDER], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


if __name__ == "__main__":
    main()
