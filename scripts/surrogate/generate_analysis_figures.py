import argparse
import json

from src.surrogate.evaluation.analysis_plots import (
    PredictionAnalysisConfig,
    generate_prediction_analysis_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate surrogate test-set analysis figures and tables.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument(
        "--prediction-path",
        default=r"data\parametric\surrogate_dataset\predictions_lightgbm.parquet",
    )
    parser.add_argument("--output-dir", default=r"outputs\surrogate\analysis_figures")
    parser.add_argument("--scatter-sample-size", type=int, default=10000)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = PredictionAnalysisConfig(
        dataset_path=args.dataset_path,
        prediction_path=args.prediction_path,
        output_dir=args.output_dir,
        scatter_sample_size=args.scatter_sample_size,
        random_seed=args.random_seed,
    )
    summary = generate_prediction_analysis_figures(cfg)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
