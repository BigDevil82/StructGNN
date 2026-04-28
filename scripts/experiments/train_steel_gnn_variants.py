from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, SAGEConv, Set2Set, global_max_pool, global_mean_pool

try:
    from torch_geometric.nn import GlobalAttention
except ImportError:
    from torch_geometric.nn import AttentionalAggregation as GlobalAttention

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
class GNNVariantConfig:
    variant: str
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    graph_repr: str = "room"
    graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    output_dir: str = r"data\parametric\ckpt\steel_gnn_variant"
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


class SteelGNNVariant(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        node_dim: int,
        edge_dim: int,
        graph_feat_dim: int,
        param_num_dim: int,
        cat_cardinalities: list[int],
        hidden_dim: int,
        gnn_layers: int,
        dropout: float,
        conv_type: str,
    ):
        super().__init__()
        if variant not in {"meanmax", "attention", "set2set", "film", "virtual_node"}:
            raise ValueError(f"Unknown variant: {variant}")
        if conv_type not in {"sage", "gine"}:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.variant = variant
        self.conv_type = conv_type
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(gnn_layers):
            if conv_type == "gine":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            else:
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.param_encoder = ParamEncoder(
            num_dim=param_num_dim,
            cat_cardinalities=cat_cardinalities,
            emb_dims=auto_param_emb_dims(cat_cardinalities),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        if variant == "attention":
            self.attn_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim // 2, 1),
                )
            )
        else:
            self.attn_pool = None

        if variant == "set2set":
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
            pooled_dim = hidden_dim * 4
        elif variant == "attention":
            self.set2set = None
            pooled_dim = hidden_dim * 3
        elif variant == "film":
            self.set2set = None
            pooled_dim = hidden_dim
            self.param_to_film = nn.Linear(hidden_dim, hidden_dim * 2)
        else:
            self.set2set = None
            pooled_dim = hidden_dim * 2

        if variant == "virtual_node":
            self.virtual_updates = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(gnn_layers)
                ]
            )
        else:
            self.virtual_updates = None

        fusion_dim = pooled_dim + (hidden_dim // 2) + graph_feat_dim + hidden_dim
        if variant == "film":
            fusion_dim = hidden_dim + (hidden_dim // 2) + graph_feat_dim + hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x = self.node_proj(data.x)
        v = None
        if self.variant == "virtual_node":
            v = global_mean_pool(x, data.batch)

        for i, (conv, bn) in enumerate(zip(self.convs, self.norms)):
            h_in = x
            if v is not None:
                h_in = h_in + v[data.batch]
            if self.conv_type == "gine":
                h = conv(h_in, data.edge_index, data.edge_attr)
            else:
                h = conv(h_in, data.edge_index)
            x = torch.relu(bn(h) + x)
            if v is not None:
                v = v + self.virtual_updates[i](global_mean_pool(x, data.batch))

        g = self._pool_nodes(x, data.batch)
        edge_h = self.edge_proj(data.edge_attr)
        edge_batch = data.batch[data.edge_index[0]]
        edge_mean = global_mean_pool(edge_h, edge_batch)
        p_emb = self.param_encoder(data.p_num, data.p_cat)

        if self.variant == "film":
            gamma, beta = self.param_to_film(p_emb).chunk(2, dim=1)
            g = g * (1.0 + gamma) + beta

        z = torch.cat([g, edge_mean, data.graph_feat, p_emb], dim=1)
        return self.head(z).squeeze(1)

    def _pool_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        g_mean = global_mean_pool(x, batch)
        g_max = global_max_pool(x, batch)
        if self.variant == "attention":
            return torch.cat([g_mean, g_max, self.attn_pool(x, batch)], dim=1)
        if self.variant == "set2set":
            return torch.cat([g_mean, g_max, self.set2set(x, batch)], dim=1)
        if self.variant == "film":
            return g_mean + g_max
        return torch.cat([g_mean, g_max], dim=1)


def run(cfg: GNNVariantConfig) -> dict[str, object]:
    set_seed(cfg.seed)
    loaders, pre, (node_dim, edge_dim, graph_feat_dim) = build_dataloaders(
        GNNDataConfig(
            dataset_path=cfg.dataset_path,
            graph_cache_dir=cfg.graph_cache_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ),
        target_col=TARGET,
    )
    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    model = SteelGNNVariant(
        variant=cfg.variant,
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_feat_dim=graph_feat_dim,
        param_num_dim=len(NUM_COLS),
        cat_cardinalities=cat_cardinalities,
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
        conv_type=cfg.conv_type,
    )
    return train_regression(
        model=model,
        loaders=loaders,
        cfg=cfg,
        label=cfg.variant,
        checkpoint_name="gnn_steel_variant.pt",
        checkpoint_payload={
            "variant": cfg.variant,
            "preprocess": pre.export(),
            "model_cfg": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "graph_feat_dim": graph_feat_dim,
                "param_num_dim": len(NUM_COLS),
                "cat_cardinalities": cat_cardinalities,
                "hidden_dim": cfg.hidden_dim,
                "gnn_layers": cfg.gnn_layers,
                "dropout": cfg.dropout,
                "conv_type": cfg.conv_type,
            },
        },
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Experimental GNN steel regression variants.")
    p.add_argument("--variant", choices=("meanmax", "attention", "set2set", "film", "virtual_node"), required=True)
    p.add_argument("--dataset-path", default=GNNVariantConfig.dataset_path)
    p.add_argument("--graph-repr", choices=("member", "room"), default="room")
    p.add_argument("--graph-cache-dir", default=GNNVariantConfig.graph_cache_dir)
    p.add_argument("--output-dir", default=GNNVariantConfig.output_dir)
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
    cfg = GNNVariantConfig(**vars(args))
    metrics = run(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
