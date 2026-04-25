import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.inference import predict_with_lightgbm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference using trained LightGBM surrogate models.")
    parser.add_argument(
        "--input-dataset",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument(
        "--artifact-dir",
        default=r"data\parametric\surrogate_dataset\baseline_lightgbm",
    )
    parser.add_argument(
        "--output-path",
        default=r"data\parametric\surrogate_dataset\predictions_lightgbm.parquet",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="Which split to run inference on. Default is test.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary classification tasks, specify or set to None to use trained model's default.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.input_dataset)
    if args.split != "all":
        if "split" not in df.columns:
            raise ValueError("Input dataset missing split column, cannot filter by split.")
        df = df[df["split"] == args.split].copy()
    pred = predict_with_lightgbm(df, args.artifact_dir, threshold=args.threshold)

    out = df[["layout_id", "sample_id"]].copy()
    out = pd.concat([out, pred], axis=1)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Rows: {len(out)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
