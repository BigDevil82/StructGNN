from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

PLOT_SENTINEL_OBJECTIVE = 1.0e10


def _is_plottable_population_point(item: dict[str, Any], value_key: str) -> bool:
    if bool(item.get("cost_preselect_skip", False)) or bool(item.get("surrogate_screen_reject", False)):
        return False
    try:
        value = float(item.get(value_key, float("nan")))
    except (TypeError, ValueError):
        return False
    return value == value and abs(value) < PLOT_SENTINEL_OBJECTIVE


def _extract_steps(history: list[dict[str, Any]]) -> list[int]:
    steps: list[int] = []
    for i, item in enumerate(history, start=1):
        if "trial" in item:
            steps.append(int(item["trial"]))
        elif "generation" in item:
            steps.append(int(item["generation"]))
        elif "iteration" in item:
            steps.append(int(item["iteration"]))
        else:
            steps.append(i)
    return steps


def _draw_objective_curve(ax, history: list[dict[str, Any]]) -> bool:
    steps = _extract_steps(history)
    current = [float(item.get("objective", item.get("best_objective", float("nan")))) for item in history]
    best = []
    best_so_far = float("inf")
    for item in history:
        v = float(item.get("best_objective", item.get("objective", float("inf"))))
        best_so_far = min(best_so_far, v)
        best.append(best_so_far)

    ax.plot(steps, current, label="current", linewidth=1.2, alpha=0.7)
    ax.plot(steps, best, label="best_so_far", linewidth=2.0)
    ax.set_xlabel("step")
    ax.set_ylabel("objective")
    ax.set_title("Optimization Objective History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return True


def _draw_feasible_ratio_curve(ax, history: list[dict[str, Any]]) -> bool:
    steps = _extract_steps(history)
    ratios: list[float] = []
    has_ratio = False
    for item in history:
        if "feasible_ratio" in item:
            ratios.append(float(item["feasible_ratio"]))
            has_ratio = True
        elif "feasible" in item:
            ratios.append(1.0 if bool(item["feasible"]) else 0.0)
        else:
            ratios.append(float("nan"))
    if not has_ratio:
        return False

    ax.plot(steps, ratios, color="tab:green", linewidth=2.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("feasible_ratio")
    ax.set_title("Feasible Ratio History")
    ax.grid(True, alpha=0.3)
    return True


def _draw_trial_scatter(ax, history: list[dict[str, Any]]) -> bool:
    xs: list[int] = []
    ys: list[float] = []
    best_line_x: list[int] = []
    best_line_y: list[float] = []
    best = float("inf")

    for item in history:
        if "trial" not in item:
            continue
        trial = int(item["trial"])
        if "objective" not in item:
            continue
        obj = float(item["objective"])
        if not (obj == obj and abs(obj) < PLOT_SENTINEL_OBJECTIVE):
            continue
        xs.append(trial)
        ys.append(obj)
        best = min(best, obj)
        best_line_x.append(trial)
        best_line_y.append(best)

    if not xs:
        return False

    ax.scatter(xs, ys, s=20, alpha=0.75, color="tab:blue", label="samples")
    ax.plot(best_line_x, best_line_y, color="tab:red", linewidth=2.0, label="best_so_far")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective")
    ax.set_title("Trial Objective Scatter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return True


def _draw_population_scatter(ax, history: list[dict[str, Any]]) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    cs: list[str] = []
    line_x: list[int] = []
    line_y: list[float] = []
    eval_count = 0

    for item in history:
        population = item.get("population")
        if not isinstance(population, list) or not population:
            continue

        obj_feasible_pairs = [
            (float(p.get("objective", float("nan"))), bool(p.get("feasible", False)))
            for p in population
            if _is_plottable_population_point(p, "objective")
        ]
        if not obj_feasible_pairs:
            continue

        for obj, feasible in obj_feasible_pairs:
            eval_count += 1
            xs.append(float(eval_count))
            ys.append(float(obj))
            cs.append("tab:green" if feasible else "tab:gray")
        line_x.append(eval_count)
        line_y.append(float(item.get("best_objective", min(obj for obj, _ in obj_feasible_pairs))))

    if not xs:
        return False

    ax.scatter(xs, ys, c=cs, s=18, alpha=0.6, label="individual")
    ax.plot(line_x, line_y, color="tab:red", linewidth=2.0, label="best_so_far")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("objective")
    ax.set_yscale("log")
    ax.set_title("Population Objective Scatter by Evaluation")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return True


def _draw_material_cost_scatter(ax, history: list[dict[str, Any]]) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    cs: list[str] = []
    eval_count = 0

    for item in history:
        population = item.get("population")
        if isinstance(population, list) and population:
            for p in population:
                if "material_cost" not in p or not _is_plottable_population_point(p, "material_cost"):
                    continue
                eval_count += 1
                xs.append(float(eval_count))
                ys.append(float(p.get("material_cost", float("nan"))))
                cs.append("tab:green" if bool(p.get("feasible", False)) else "tab:gray")
            continue

        if "trial" in item:
            objectives = item.get("objectives", {})
            if isinstance(objectives, dict) and "material_cost" in objectives:
                eval_count += 1
                xs.append(float(eval_count))
                ys.append(float(objectives["material_cost"]))
                cs.append("tab:green" if bool(item.get("feasible", False)) else "tab:gray")

    if not xs:
        return False

    ax.scatter(xs, ys, c=cs, s=20, alpha=0.7)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("material_cost")
    ax.set_title("Material Cost Scatter by Evaluation")
    ax.grid(True, alpha=0.3)
    return True


def _draw_pareto_scatter(ax, history: list[dict[str, Any]]) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    has_pareto = False
    y_label = "total_violation"

    for item in history:
        second_key = "total_violation" if "total_violation" in item else "margin_penalty"
        if "material_cost" not in item or second_key not in item:
            continue
        xs.append(float(item["material_cost"]))
        ys.append(float(item[second_key]))
        y_label = second_key
        pareto = bool(item.get("pareto", False))
        has_pareto = has_pareto or pareto
        colors.append("tab:red" if pareto else "tab:blue")

    if not xs:
        return False

    ax.scatter(xs, ys, c=colors, s=24, alpha=0.85)
    ax.set_xlabel("material_cost")
    ax.set_ylabel(y_label)
    title = "Pareto Scatter"
    if has_pareto:
        title += " (red = pareto)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return True


def _draw_empty_panel(ax, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def generate_optimization_plots(
    payload: dict[str, Any],
    result_path: str | Path,
    out_dir: str | Path | None = None,
) -> list[str]:
    history = list(payload.get("history", []))
    if not history:
        return []

    result_path = Path(result_path)
    plot_dir = Path(out_dir) if out_dir else result_path.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    algorithm = str(payload.get("algorithm", "")).lower()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    panels = axes.ravel()

    _draw_objective_curve(panels[0], history)

    if not _draw_population_scatter(panels[1], history):
        if not _draw_trial_scatter(panels[1], history):
            _draw_empty_panel(panels[1], "Evaluation Objective Scatter")

    if algorithm == "nsga2":
        if not _draw_pareto_scatter(panels[2], history):
            _draw_empty_panel(panels[2], "Pareto Scatter")
    elif not _draw_feasible_ratio_curve(panels[2], history):
        _draw_empty_panel(panels[2], "Feasible Ratio History")

    if not _draw_material_cost_scatter(panels[3], history):
        _draw_empty_panel(panels[3], "Material Cost Scatter by Evaluation")

    fig.suptitle(f"Optimization Summary: {result_path.stem}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = plot_dir / f"{result_path.stem}_optimization_summary.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return [str(out_path)]
