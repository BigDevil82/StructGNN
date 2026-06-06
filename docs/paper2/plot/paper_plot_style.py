from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PLOT_DIR = ROOT / "outputs/result/optimization/plots"

METHOD_ORDER = ["full", "gnn_cost", "gnn_screen_cost"]
SURROGATE_METHODS = ["gnn_cost", "gnn_screen_cost"]
METHOD_LABELS = {
    "full": "Full FEA",
    "gnn_cost": "Cost surrogate",
    "gnn_screen_cost": "Screen + cost",
}
METHOD_LABELS_SHORT = {
    "full": "Full FEA",
    "gnn_cost": "Cost",
    "gnn_screen_cost": "Screen + cost",
}
ALGORITHM_ORDER = ["GA", "PSO", "Random Search"]
METHOD_COLORS = {
    "full": "#8fb6d6",
    "gnn_cost": "#efbd75",
    "gnn_screen_cost": "#e8a69d",
}
FAMILY_MARKERS = {"L17": "o", "L27": "s", "L1L28": "^"}


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#222222",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, filename: str, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else PLOT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[plot] saved: {path}")
    return path


def draw_violin_points(
    ax: plt.Axes,
    data: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    ylabel: str | None = None,
    title: str | None = None,
    jitter_seed: int = 20260606,
    width: float = 0.78,
    point_size: float = 8.0,
) -> None:
    positions = np.arange(1, len(data) + 1)
    clean = [np.asarray(values, dtype=float) for values in data]
    parts = ax.violinplot(
        clean,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.72)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(jitter_seed)
    for x, values, color in zip(positions, clean, colors):
        if len(values) == 0:
            continue
        jitter = rng.uniform(-0.085, 0.085, size=len(values))
        ax.scatter(
            x + jitter,
            values,
            s=point_size,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.3,
            alpha=0.35,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )


def family_of(layout_id: str) -> str:
    if str(layout_id).startswith("L1L28_"):
        return "L1L28"
    return str(layout_id).split("_", 1)[0]


def method_colors(methods: Iterable[str]) -> list[str]:
    return [METHOD_COLORS[method] for method in methods]
