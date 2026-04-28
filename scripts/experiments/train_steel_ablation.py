from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.gnn.dataset import CAT_COLS, NUM_COLS, GNNDataConfig, build_dataloaders
from src.surrogate.gnn.model import ParamEncoder, auto_param_emb_dims

TARGET = "material_steel_kg"


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
        z = self.param_encoder(data.p_num, data.p_cat)
        return self.head(z).squeeze(1)


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
        z = torch.cat([p, data.graph_feat], dim=1)
        return self.head(z).squeeze(1)


def run(cfg: SteelAblationConfig) -> dict[str, object]:
    _set_seed(cfg.seed)
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

    y_mean, y_std = _target_stats(loaders["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        train_loss = _train_one_epoch(model, loaders["train"], criterion, optimizer, device, y_mean, y_std)
        val = _eval(model, loaders["val"], device, y_mean, y_std)
        score = float(val["mae"])

        if epoch == 0 or (epoch + 1) % max(1, cfg.log_interval) == 0:
            print(
                f"[steel-ablation][{cfg.baseline}] epoch={epoch + 1}/{cfg.max_epochs} "
                f"train_loss={train_loss:.6f} val_mae={val['mae']:.3f} "
                f"val_rmse={val['rmse']:.3f} val_r2={val['r2']:.6f} best_mae={best_score:.3f}"
            )

        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                print(
                    f"[steel-ablation][{cfg.baseline}] early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch + 1} best_mae={best_score:.3f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")

    model.load_state_dict(best_state)
    val = _eval(model, loaders["val"], device, y_mean, y_std)
    test = _eval(model, loaders["test"], device, y_mean, y_std)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "baseline": cfg.baseline,
            "state_dict": model.cpu().state_dict(),
            "preprocess": pre.export(),
            "target": TARGET,
            "target_transform": "standardized_log1p",
            "target_log_mean": float(y_mean),
            "target_log_std": float(y_std),
            "cat_cardinalities": cat_cardinalities,
            "graph_feat_dim": int(graph_feat_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
        },
        out_dir / "steel_ablation.pt",
    )

    test_df = pd.read_parquet(cfg.dataset_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = test_df[["layout_id", "sample_id", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "steel_true_kg"})
    pred_df["steel_pred_kg"] = test["y_pred"]
    pred_df.to_parquet(out_dir / "predictions_test.parquet", index=False)

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "target_log_mean": float(y_mean),
        "target_log_std": float(y_std),
        "val": _drop_arrays(val),
        "test": _drop_arrays(test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


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


def _target_stats(loader) -> tuple[float, float]:
    vals = []
    for batch in loader:
        vals.append(batch.y.detach().cpu().numpy())
    y = np.log1p(np.concatenate(vals, axis=0).astype(np.float64))
    return float(y.mean()), float(max(y.std(ddof=0), 1.0e-6))


def _train_one_epoch(model, loader, criterion, optimizer, device: torch.device, y_mean: float, y_std: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        y = _normalize_target(batch.y, y_mean, y_std)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _eval(model, loader, device: torch.device, y_mean: float, y_std: float) -> dict[str, object]:
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            pred = _denormalize_target(pred_norm, y_mean, y_std)
            ys.append(batch.y.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).astype(np.float64)
    y_pred = np.maximum(np.concatenate(preds, axis=0).astype(np.float64), 0.0)
    return {
        "samples": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1.0))),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def _normalize_target(y: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return (torch.log1p(y) - y_mean) / y_std


def _denormalize_target(y: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return torch.expm1(y * y_std + y_mean)


def _drop_arrays(stat: dict[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in stat.items() if k not in {"y_true", "y_pred"}}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
