import argparse
from pathlib import Path

import pandas as pd

from src.surrogate.features.layout_features import (
    LayoutFeatureConfig,
    extract_layout_features,
    merge_layout_features,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract layout statistical features and merge with surrogate sample table."
    )
    parser.add_argument("--layout-dir", default=r"data\dxf\cad_json_data\fem_raw")
    parser.add_argument(
        "--sample-dataset", default=r"data\parametric\surrogate_dataset\surrogate_samples.parquet"
    )
    parser.add_argument(
        "--output-feature-path",
        default=r"data\parametric\surrogate_dataset\layout_features.parquet",
    )
    parser.add_argument(
        "--output-merged-path",
        default=r"data\parametric\surrogate_dataset\surrogate_samples_with_layout_features.parquet",
    )
    parser.add_argument("--xy-scale-to-m", type=float, default=0.001)
    parser.add_argument("--boundary-band-ratio", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = LayoutFeatureConfig(
        layout_dir=args.layout_dir,
        xy_scale_to_m=args.xy_scale_to_m,
        boundary_band_ratio=args.boundary_band_ratio,
    )

    feature_df = extract_layout_features(cfg)
    out_feature = Path(args.output_feature_path)
    out_feature.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_feature, index=False)

    sample_df = pd.read_parquet(args.sample_dataset)
    merged = merge_layout_features(sample_df, feature_df)
    out_merged = Path(args.output_merged_path)
    out_merged.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_merged, index=False)

    print(f"Layouts: {feature_df['layout_id'].nunique()}")
    print(f"Layout features: {len(feature_df.columns) - 1}")
    print(f"Merged rows: {len(merged)}")
    print(f"Feature file: {out_feature}")
    print(f"Merged file: {out_merged}")


if __name__ == "__main__":
    main()
