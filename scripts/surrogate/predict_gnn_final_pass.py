import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.gnn.inference import predict_with_gnn


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict final_pass with trained surrogate GNN model.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--artifact-path",
        default=r"data\parametric\surrogate_dataset\baseline_gnn_final_pass\gnn_final_pass.pt",
    )
    parser.add_argument("--graph-cache-dir", default=r"data\parametric\surrogate_dataset\gnn_graph_cache")
    parser.add_argument(
        "--output-path",
        default=r"data\parametric\surrogate_dataset\predictions_gnn_final_pass.parquet",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--screening-threshold", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    df = pd.read_parquet(args.dataset_path)
    if args.split:
        df = df[df["split"] == args.split].copy().reset_index(drop=True)

    pred = predict_with_gnn(
        df,
        artifact_path=args.artifact_path,
        graph_cache_dir=args.graph_cache_dir,
        threshold=args.threshold,
        screening_threshold=args.screening_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        report_metrics=True,
    )

    out = df[["layout_id", "sample_id", "final_pass"]].copy()
    out = out.rename(columns={"final_pass": "final_pass_true"})
    out = pd.concat([out, pred.reset_index(drop=True)], axis=1)
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    print(f"Rows: {len(out)}")
    print(f"Output: {args.output_path}")


if __name__ == "__main__":
    main()
