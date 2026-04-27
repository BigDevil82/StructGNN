import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.training.lightgbm_steel_quantile import SteelQuantileConfig, train_steel_quantile_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LightGBM quantile model for steel usage.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset\steel_quantile_lightgbm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--upper-alpha", type=float, default=0.95)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = SteelQuantileConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        upper_alpha=args.upper_alpha,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    metrics = train_steel_quantile_model(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
