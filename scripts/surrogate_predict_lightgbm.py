import argparse
from pathlib import Path

import pandas as pd

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_parquet(args.input_dataset)
    pred = predict_with_lightgbm(df, args.artifact_dir)

    out = df[["layout_id", "sample_id"]].copy()
    out = pd.concat([out, pred], axis=1)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"Rows: {len(out)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
