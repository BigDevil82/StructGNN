import argparse
import json

from src.surrogate.data.split import GroupSplitConfig, build_group_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create layout-grouped train/val/test and GroupKFold split indices for surrogate dataset."
    )
    parser.add_argument(
        "--sample-dataset",
        default=r"data\parametric\surrogate_dataset\surrogate_samples_with_layout_features.parquet",
    )
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset\splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = GroupSplitConfig(
        sample_dataset_path=args.sample_dataset,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        n_splits=args.n_splits,
    )
    summary = build_group_splits(cfg)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
