from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from steel_regression_common import TARGET, set_seed, train_regression
from src.surrogate.gnn.dataset import CAT_COLS, NUM_COLS, GNNDataConfig, build_dataloaders
from src.surrogate.gnn.model import ParamEncoder, auto_param_emb_dims


@dataclass(frozen=True)
class SteelAblationConfig:
    baseline: str
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    graph_repr: str = "room"
    graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    output_dir: str = r"data\parametric\ckpt\steel_ablation"
    seed: int = 42
    batch_size: int = 512
    max_epochs: int = 2
    early_stop_rounds: int = 1
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    gnn_layers: int = 3
    conv_type: str = "sage"
    dropout: float = 0.1
    num_workers: int = 0
    log_interval: int = 1


class ParamOnlySteelMLP(nn.Module):
    def __init__(self, param_num_dim: int, cat_cardinalities: list[int], hidden_dim: int, dropout: float):
        super().__init__()
        self.param_encoder = ParamEncoder(
            num_dim=param_num_dim,
            cat_cardinalities=cat_cardinalities,
            emb_dims=auto_param_emb_dims(cat_cardinalities),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = _head(hidden_dim, hidden_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        return self.head(self.param_encoder(data.p_num, data.p_cat)).squeeze(1)


class ParamGraphFeatSteelMLP(nn.Module):
    def __init__(
        self,
        graph_feat_dim: int,
        param_num_dim: int,
        cat_cardinalities: list[int],
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.param_encoder = ParamEncoder(
            num_dim=param_num_dim,
            cat_cardinalities=cat_cardinalities,
            emb_dims=auto_param_emb_dims(cat_cardinalities),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = _head(hidden_dim + graph_feat_dim, hidden_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        p = self.param_encoder(data.p_num, data.p_cat)
        return self.head(torch.cat([p, data.graph_feat], dim=1)).squeeze(1)


def run(cfg: SteelAblationConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    loaders, pre, (_, _, graph_feat_dim) = build_dataloaders(
        GNNDataConfig(
            dataset_path=cfg.dataset_path,
            graph_cache_dir=cfg.graph_cache_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ),
        target_col=TARGET,
    )

    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    if cfg.baseline == "param_only":
        model = ParamOnlySteelMLP(len(NUM_COLS), cat_cardinalities, cfg.hidden_dim, cfg.dropout)
    elif cfg.baseline == "param_graph_feat":
        model = ParamGraphFeatSteelMLP(graph_feat_dim, len(NUM_COLS), cat_cardinalities, cfg.hidden_dim, cfg.dropout)
    else:
        raise ValueError(f"Unknown baseline: {cfg.baseline}")

    return train_regression(
        model=model,
        loaders=loaders,
        cfg=cfg,
        label=cfg.baseline,
        checkpoint_name="steel_ablation.pt",
        checkpoint_payload={
            "baseline": cfg.baseline,
            "preprocess": pre.export(),
            "cat_cardinalities": cat_cardinalities,
            "graph_feat_dim": int(graph_feat_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Steel regression ablations outside production src code.")
    p.add_argument("--baseline", choices=("param_only", "param_graph_feat"), required=True)
    p.add_argument("--dataset-path", default=SteelAblationConfig.dataset_path)
    p.add_argument("--graph-repr", choices=("member", "room"), default="room")
    p.add_argument("--graph-cache-dir", default=SteelAblationConfig.graph_cache_dir)
    p.add_argument("--output-dir", default=SteelAblationConfig.output_dir)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=2)
    p.add_argument("--early-stop-rounds", type=int, default=1)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=3)
    p.add_argument("--conv-type", choices=("sage", "gine"), default="sage")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=1)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = SteelAblationConfig(**vars(args))
    metrics = run(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


def _head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )


if __name__ == "__main__":
    main()
