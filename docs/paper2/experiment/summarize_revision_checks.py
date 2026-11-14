"""Recompute manuscript review statistics from saved results; no model fitting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "docs/paper2/plot"))

from optimization_plot_data import SELECTED15_EXPERIMENTS, load_all_summaries, paired_with_full


def main() -> None:
    data = load_all_summaries(SELECTED15_EXPERIMENTS)
    paired = paired_with_full(data)
    screen = paired[paired["method"] == "gnn_screen_cost"]
    cost = screen["cost_gap_pct"].dropna()
    report = {
        "optimization_sources": {
            name: str(path.relative_to(ROOT)) for name, path in SELECTED15_EXPERIMENTS.items()
        },
        "full_feasible_cases": data[data["method"] == "full"]
        .groupby("algorithm")["best_feasible"].sum().to_dict(),
        "paired_cost_by_algorithm": screen.groupby("algorithm")["cost_gap_pct"]
        .agg(["count", "mean", "median", "max"]).to_dict(orient="index"),
        "pooled_cost_change_pct": cost.agg(["count", "mean", "median", "max"]).to_dict(),
        "mean_paired_fea_reduction_pct": float(100 * screen["fea_reduction"].mean()),
        "fea_reduction_runs": len(screen),
    }

    path = ROOT / "outputs/result/paper2/local_calibration/feasibility/results.csv"
    feas = pd.read_csv(path)
    feas = feas[feas["layout_id"] != "L17_157"]
    metrics = ["brier", "ece", "screen_recall", "screen_reject_rate"]
    report["platt_by_sample_count"] = (
        feas[feas["method"] == "platt"].groupby("calib_n")[metrics].mean().to_dict(orient="index")
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
