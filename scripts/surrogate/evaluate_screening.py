import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate surrogate models as feasibility pre-screeners.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--val-pred-path", required=True)
    parser.add_argument("--test-pred-path", required=True)
    parser.add_argument("--prob-col", default="pred_final_pass_prob")
    parser.add_argument("--target-recalls", nargs="+", type=float, default=[0.99, 0.98, 0.95, 0.9])
    parser.add_argument("--threshold-step", type=float, default=0.002)
    parser.add_argument("--output-json", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base = pd.read_parquet(args.dataset_path, columns=["layout_id", "sample_id", "split", "final_pass"])
    val = _load_split_predictions(base, args.val_pred_path, args.prob_col, "val")
    test = _load_split_predictions(base, args.test_pred_path, args.prob_col, "test")

    result = {
        "val_pred_path": args.val_pred_path,
        "test_pred_path": args.test_pred_path,
        "prob_col": args.prob_col,
        "targets": {},
    }

    for target in args.target_recalls:
        threshold, val_metrics = _select_threshold(val, target, args.threshold_step)
        test_metrics = _screening_metrics(test, threshold)
        result["targets"][f"{target:.3f}"] = {
            "threshold": threshold,
            "val": val_metrics,
            "test": test_metrics,
        }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


def _load_split_predictions(base: pd.DataFrame, pred_path: str, prob_col: str, split: str) -> pd.DataFrame:
    keys = base[base["split"] == split][["layout_id", "sample_id", "final_pass"]].copy()
    pred = pd.read_parquet(pred_path)
    if prob_col not in pred.columns:
        raise ValueError(f"Prediction file missing probability column: {prob_col}")

    cols = ["layout_id", "sample_id", prob_col]
    out = keys.merge(pred[cols], on=["layout_id", "sample_id"], how="inner", validate="one_to_one")
    if len(out) != len(keys):
        raise ValueError(f"{split} predictions do not cover the split: {len(out)} / {len(keys)}")
    return out.rename(columns={prob_col: "prob"})


def _select_threshold(df: pd.DataFrame, target_recall: float, step: float) -> tuple[float, dict[str, float]]:
    best_threshold = None
    best_metrics = None
    for threshold in np.arange(0.0, 1.0 + step * 0.5, step):
        metrics = _screening_metrics(df, float(threshold))
        if metrics["feasible_recall"] < target_recall:
            continue
        if best_metrics is None or metrics["reject_rate"] > best_metrics["reject_rate"]:
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise ValueError(f"No threshold satisfies target feasible recall: {target_recall}")
    return best_threshold, best_metrics


def _screening_metrics(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = df["final_pass"].astype(bool).to_numpy()
    reject = df["prob"].to_numpy(dtype=float) < threshold
    keep = ~reject

    feasible = y
    infeasible = ~y
    feasible_count = max(int(feasible.sum()), 1)
    reject_count = max(int(reject.sum()), 1)
    keep_count = max(int(keep.sum()), 1)

    reject_feasible = reject & feasible
    reject_infeasible = reject & infeasible
    kept_feasible = keep & feasible
    kept_infeasible = keep & infeasible

    return {
        "samples": float(len(df)),
        "threshold": float(threshold),
        "reject_rate": float(reject.mean()),
        "keep_rate": float(keep.mean()),
        "feasible_recall": float(kept_feasible.sum() / feasible_count),
        "false_reject_rate": float(reject_feasible.sum() / feasible_count),
        "false_reject_count": float(reject_feasible.sum()),
        "reject_infeasible_precision": float(reject_infeasible.sum() / reject_count),
        "kept_feasible_rate": float(kept_feasible.sum() / keep_count),
        "kept_infeasible_rate": float(kept_infeasible.sum() / keep_count),
    }


if __name__ == "__main__":
    main()
