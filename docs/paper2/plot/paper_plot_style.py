from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
PLOT_DIR = ROOT / "outputs/result/paper2/all_figs"

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
PANEL_CAPTION_SIZE = 11
OPTIMA_BOLD_PATH = Path.home() / r"AppData\Local\Microsoft\Windows\Fonts\OPTIMA BOLD.TTF"


def register_paper_fonts() -> None:
    font_paths = [
        Path.home() / r"AppData\Local\Microsoft\Windows\Fonts\OPTIMA.OTF",
        OPTIMA_BOLD_PATH,
        Path(r"C:\Windows\Fonts\OPTIMA.OTF"),
        Path(r"C:\Windows\Fonts\OPTIMA BOLD.TTF"),
    ]
    for path in font_paths:
        if path.exists():
            font_manager.fontManager.addfont(str(path))


def set_paper_style() -> None:
    register_paper_fonts()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Optima",
                "Optima LT Std",
                "Gill Sans",
                "Segoe UI",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.fontsize": 11,
            "figure.titlesize": 13,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "#222222",
            "grid.color": "#d9d9d9",
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def bold_font(size: float | None = None) -> FontProperties:
    register_paper_fonts()
    if OPTIMA_BOLD_PATH.exists():
        return FontProperties(fname=str(OPTIMA_BOLD_PATH), size=size)
    return FontProperties(family="sans-serif", weight="bold", size=size)


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


def family_of(layout_id: str) -> str:
    if str(layout_id).startswith("L1L28_"):
        return "L1L28"
    return str(layout_id).split("_", 1)[0]


def method_colors(methods: Iterable[str]) -> list[str]:
    return [METHOD_COLORS[method] for method in methods]
