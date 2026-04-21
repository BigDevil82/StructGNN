import argparse
import json

from src.surrogate.data import SurrogateDataPrepConfig, prepare_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare surrogate training dataset from parametric analysis parquet files."
    )
    parser.add_argument("--source-dir", default=r"data\parametric\parametric_dataset_5000x143")
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset")
    parser.add_argument("--file-glob", default="*.parquet")
    parser.add_argument("--drop-non-converged", action="store_true")
    parser.add_argument("--disable-deduplicate", action="store_true")
    parser.add_argument("--drift-limit", type=float, default=1.0 / 1000.0)
    parser.add_argument("--torsion-limit", type=float, default=1.5)
    parser.add_argument("--stiffness-limit", type=float, default=0.7)
    parser.add_argument("--default-shear-limit", type=float, default=0.016)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SurrogateDataPrepConfig(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        file_glob=args.file_glob,
        drop_non_converged=args.drop_non_converged,
        deduplicate=not args.disable_deduplicate,
        drift_limit=args.drift_limit,
        torsion_limit=args.torsion_limit,
        stiffness_limit=args.stiffness_limit,
        default_shear_limit=args.default_shear_limit,
    )
    summary = prepare_training_dataset(cfg)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
