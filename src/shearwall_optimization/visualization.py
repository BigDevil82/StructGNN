from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


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


def _plot_objective_curve(history: list[dict[str, Any]], out_path: Path) -> None:
    steps = _extract_steps(history)
    current = [float(item.get("objective", item.get("best_objective", float("nan")))) for item in history]
    best = []
    best_so_far = float("inf")
    for item in history:
        v = float(item.get("best_objective", item.get("objective", float("inf"))))
        best_so_far = min(best_so_far, v)
        best.append(best_so_far)

    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, current, label="current", linewidth=1.2, alpha=0.7)
    plt.plot(steps, best, label="best_so_far", linewidth=2.0)
    plt.xlabel("step")
    plt.ylabel("objective")
    plt.title("Optimization Objective History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_feasible_ratio_curve(history: list[dict[str, Any]], out_path: Path) -> bool:
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

    plt.figure(figsize=(8, 4.8))
    plt.plot(steps, ratios, color="tab:green", linewidth=2.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("step")
    plt.ylabel("feasible_ratio")
    plt.title("Feasible Ratio History")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _plot_trial_scatter(history: list[dict[str, Any]], out_path: Path) -> bool:
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
        xs.append(trial)
        ys.append(obj)
        best = min(best, obj)
        best_line_x.append(trial)
        best_line_y.append(best)

    if not xs:
        return False

    plt.figure(figsize=(8, 4.8))
    plt.scatter(xs, ys, s=20, alpha=0.75, color="tab:blue", label="samples")
    plt.plot(best_line_x, best_line_y, color="tab:red", linewidth=2.0, label="best_so_far")
    plt.xlabel("trial")
    plt.ylabel("objective")
    plt.title("Trial Objective Scatter")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _plot_population_scatter(history: list[dict[str, Any]], out_path: Path) -> bool:
    xs: list[float] = []
    ys: list[float] = []
    cs: list[str] = []
    line_x: list[int] = []
    line_y: list[float] = []
    eval_count = 0

    for item in history:
        objs = item.get("population_objectives")
        feas = item.get("population_feasible")
        if not isinstance(objs, list) or not objs:
            continue
        if not isinstance(feas, list) or len(feas) != len(objs):
            feas = [False for _ in objs]
        for j, obj in enumerate(objs):
            eval_count += 1
            xs.append(float(eval_count))
            ys.append(float(obj))
            cs.append("tab:green" if bool(feas[j]) else "tab:gray")
        line_x.append(eval_count)
        line_y.append(float(item.get("best_objective", min(float(v) for v in objs))))

    if not xs:
        return False

    plt.figure(figsize=(9, 5.2))
    plt.scatter(xs, ys, c=cs, s=18, alpha=0.6, label="individual")
    plt.plot(line_x, line_y, color="tab:red", linewidth=2.0, label="best_so_far")
    plt.xlabel("evaluation")
    plt.ylabel("objective")
    plt.title("Population Objective Scatter by Evaluation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


def _plot_pareto_scatter(history: list[dict[str, Any]], out_path: Path) -> bool:
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

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(xs, ys, c=colors, s=24, alpha=0.85)
    plt.xlabel("material_cost")
    plt.ylabel(y_label)
    title = "Pareto Scatter"
    if has_pareto:
        title += " (red = pareto)"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True


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
    saved: list[str] = []

    obj_path = plot_dir / f"{result_path.stem}_objective.png"
    _plot_objective_curve(history, obj_path)
    saved.append(str(obj_path))

    pop_path = plot_dir / f"{result_path.stem}_population_scatter.png"
    if _plot_population_scatter(history, pop_path):
        saved.append(str(pop_path))

    trial_path = plot_dir / f"{result_path.stem}_trial_scatter.png"
    if _plot_trial_scatter(history, trial_path):
        saved.append(str(trial_path))

    feasible_path = plot_dir / f"{result_path.stem}_feasible_ratio.png"
    if _plot_feasible_ratio_curve(history, feasible_path):
        saved.append(str(feasible_path))

    pareto_path = plot_dir / f"{result_path.stem}_pareto.png"
    if algorithm == "nsga2" and _plot_pareto_scatter(history, pareto_path):
        saved.append(str(pareto_path))

    return saved
