import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize optimization result JSON.")
    p.add_argument("--result-json", required=True)
    p.add_argument("--out-dir", default="")
    return p


def _extract_steps(history: list[dict]) -> list[int]:
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


def _plot_objective_curve(history: list[dict], out_path: Path) -> None:
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


def _plot_pareto_scatter(history: list[dict], out_path: Path) -> bool:
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


def main() -> None:
    args = build_parser().parse_args()
    result_path = Path(args.result_json)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    history = list(payload.get("history", []))
    if not history:
        raise ValueError("Result JSON has empty history.")

    out_dir = Path(args.out_dir) if args.out_dir else result_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_path = out_dir / f"{result_path.stem}_objective.png"
    _plot_objective_curve(history, obj_path)

    pareto_path = out_dir / f"{result_path.stem}_pareto.png"
    pareto_ok = _plot_pareto_scatter(history, pareto_path)

    print(f"saved: {obj_path}")
    if pareto_ok:
        print(f"saved: {pareto_path}")
    else:
        print("skip: pareto plot (no material_cost/total_violation in history)")


if __name__ == "__main__":
    main()
