import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.gnn.train import GNNTrainConfig, run_gnn_train

DEFAULT_GRAPH_CACHE_DIR = r"data\parametric\surrogate_dataset\gnn_graph_cache"
DEFAULT_ROOM_GRAPH_CACHE_DIR = r"data\parametric\surrogate_dataset\gnn_room_graph_cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train GNN classifier for surrogate final_pass.")
    parser.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    parser.add_argument("--layout-json-dir", default=r"data\dxf\cad_json_data\fem_raw")
    parser.add_argument("--layout-dxf-dir", default=r"data\dxf\fem_raw")
    parser.add_argument("--graph-repr", choices=("member", "room"), default="member")
    parser.add_argument("--graph-cache-dir", default=None)
    parser.add_argument("--output-dir", default=r"data\parametric\surrogate_dataset\baseline_gnn_final_pass")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--early-stop-rounds", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=3)
    parser.add_argument("--conv-type", choices=("sage", "gine"), default="sage")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--monitor-metric", choices=("pr_auc", "f1"), default="pr_auc")
    parser.add_argument("--rebuild-graph-cache", action="store_true")
    parser.add_argument("--no-merge-members", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph_cache_dir = args.graph_cache_dir
    if graph_cache_dir is None:
        graph_cache_dir = DEFAULT_ROOM_GRAPH_CACHE_DIR if args.graph_repr == "room" else DEFAULT_GRAPH_CACHE_DIR

    cfg = GNNTrainConfig(
        dataset_path=args.dataset_path,
        layout_json_dir=args.layout_json_dir,
        layout_dxf_dir=args.layout_dxf_dir,
        graph_cache_dir=graph_cache_dir,
        graph_repr=args.graph_repr,
        output_dir=args.output_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stop_rounds=args.early_stop_rounds,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        conv_type=args.conv_type,
        dropout=args.dropout,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        monitor_metric=args.monitor_metric,
        rebuild_graph_cache=args.rebuild_graph_cache,
        merge_members=not args.no_merge_members,
    )
    metrics = run_gnn_train(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
