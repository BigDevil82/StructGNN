from __future__ import annotations

import argparse
from pathlib import Path

from optimization_plot_data import DEFAULT_EXPERIMENTS, load_summary
from paper_plot_style import (
    ALGORITHM_ORDER,
    METHOD_LABELS_SHORT,
    METHOD_ORDER,
    PLOT_DIR,
    draw_violin_points,
    method_colors,
    save_figure,
    set_paper_style,
)

import matplotlib.pyplot as plt


def main() -> None:
    args = build_parser().parse_args()
    set_paper_style()
    experiments = {
        "GA": Path(args.ga_dir),
        "PSO": Path(args.pso_dir),
        "Random Search": Path(args.random_dir),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    y_max = 0.0
    for ax, name in zip(axes, ALGORITHM_ORDER):
        df = load_summary(experiments[name])
        data = [df.loc[df["method"] == method, "fea_calls"].dropna().to_numpy() for method in METHOD_ORDER]
        y_max = max(y_max, max((values.max() for values in data if len(values)), default=0.0))
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
            method_colors(METHOD_ORDER),
            ylabel="FEA calls" if ax is axes[0] else None,
            title=name,
            point_size=5,
        )

    for ax in axes:
        ax.set_ylim(0, y_max * 1.08)
    fig.suptitle("FEA Calls by Optimization Algorithm and Surrogate Strategy", y=1.02)
    fig.tight_layout()
    save_figure(fig, args.filename, args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot violin distributions of FEA calls for optimization experiments.")
    p.add_argument("--ga-dir", default=str(DEFAULT_EXPERIMENTS["GA"]))
    p.add_argument("--pso-dir", default=str(DEFAULT_EXPERIMENTS["PSO"]))
    p.add_argument("--random-dir", default=str(DEFAULT_EXPERIMENTS["Random Search"]))
    p.add_argument("--out-dir", default=str(PLOT_DIR))
    p.add_argument("--filename", default="fea_calls_violin.png")
    return p


if __name__ == "__main__":
    main()
