import argparse
import json

from src.surrogate.training.catboost_baseline import CatBoostBaselineConfig, run_catboost_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CatBoost baseline for surrogate modeling core tasks.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset\baseline_catboost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = CatBoostBaselineConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    metrics = run_catboost_baseline(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
