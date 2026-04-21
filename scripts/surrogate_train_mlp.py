import argparse
import json

from src.surrogate.training.mlp_classifier import MLPClassifierConfig, run_mlp_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train first-version MLP classifier for final_pass prediction."
    )
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset\baseline_mlp_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stop-rounds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-dims", nargs=3, type=int, default=[256, 128, 64])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = MLPClassifierConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_rounds=args.early_stop_rounds,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    )
    metrics = run_mlp_classifier(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
