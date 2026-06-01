from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.surrogate.gnn.dataset import CAT_COLS, NUM_COLS, GNNDataConfig, build_dataloaders
from src.surrogate.gnn.graph_data import (
    ROOM_GRAPH_FEATURE_VERSION,
    LayoutGraphCacheConfig,
    build_layout_graph_cache,
)
from src.surrogate.gnn.model import LayoutParamGNN, auto_param_emb_dims

TARGET = "material_steel_kg"


@dataclass(frozen=True)
class GNNSteelTrainConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    layout_json_dir: str = r"data\dxf\cad_json_data\fem_raw"
    layout_dxf_dir: str = r"data\dxf\fem_raw"
    graph_cache_dir: str = r"data\parametric\surrogate_dataset\gnn_room_graph_cache"
    graph_repr: str = "room"
    output_dir: str = r"data\parametric\surrogate_dataset\steel_gnn_room"
    seed: int = 42
    batch_size: int = 128
    max_epochs: int = 50
    early_stop_rounds: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    gnn_layers: int = 3
    conv_type: str = "sage"
    dropout: float = 0.15
    num_workers: int = 0
    log_interval: int = 1
    rebuild_graph_cache: bool = False
    merge_members: bool = True


def run_gnn_steel_train(cfg: GNNSteelTrainConfig) -> dict[str, object]:
    _set_seed(cfg.seed)

    if cfg.rebuild_graph_cache or _needs_graph_cache_rebuild(cfg):
        build_layout_graph_cache(
            LayoutGraphCacheConfig(
                layout_json_dir=cfg.layout_json_dir,
                layout_dxf_dir=cfg.layout_dxf_dir,
                output_dir=cfg.graph_cache_dir,
                graph_repr=cfg.graph_repr,
                merge_members=cfg.merge_members,
            )
        )

    loaders, pre, (node_dim, edge_dim, graph_feat_dim) = build_dataloaders(
        GNNDataConfig(
            dataset_path=cfg.dataset_path,
            graph_cache_dir=cfg.graph_cache_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ),
        target_col=TARGET,
    )

    y_mean, y_std = _target_stats(loaders["train"])
    model = _build_model(cfg, pre, node_dim, edge_dim, graph_feat_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(1, cfg.max_epochs + 1):
        # epoch is 1-based for logging
        train_loss = _train_one_epoch(model, loaders["train"], criterion, optimizer, device, y_mean, y_std)
        val_stat = _eval_regression(model, loaders["val"], device, y_mean, y_std)
        monitor = float(val_stat["mae"])
        scheduler.step(monitor)

        if monitor < best_score:
            best_score = monitor
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                print(
                    f"[surrogate][gnn-steel] early_stop epoch={epoch} "
                    f"best_epoch={best_epoch} best_mae={best_score:.3f}"
                )
                break

        if epoch == 1 or (epoch) % max(cfg.log_interval, 1) == 0:
            print(
                f"[surrogate][gnn-steel] epoch={epoch}/{cfg.max_epochs} "
                f"train_loss={train_loss:.6f} val_mae={val_stat['mae']:.3f} "
                f"val_rmse={val_stat['rmse']:.3f} val_r2={val_stat['r2']:.6f} best_mae={best_score:.3f}"
            )

    if best_state is None:
        raise RuntimeError("GNN steel model failed to produce a valid checkpoint")

    model.load_state_dict(best_state)
    val_stat = _eval_regression(model, loaders["val"], device, y_mean, y_std)
    test_stat = _eval_regression(model, loaders["test"], device, y_mean, y_std)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    emb_dims = auto_param_emb_dims(cat_cardinalities)

    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "model_cfg": {
                "node_dim": node_dim,
                "edge_dim": edge_dim,
                "graph_feat_dim": graph_feat_dim,
                "param_num_dim": len(NUM_COLS),
                "param_cat_cardinalities": cat_cardinalities,
                "param_emb_dims": emb_dims,
                "hidden_dim": cfg.hidden_dim,
                "gnn_layers": cfg.gnn_layers,
                "dropout": cfg.dropout,
                "conv_type": cfg.conv_type,
            },
            "preprocess": pre.export(),
            "target": TARGET,
            "target_transform": "standardized_log1p",
            "target_log_mean": float(y_mean),
            "target_log_std": float(y_std),
        },
        out_dir / "gnn_steel.pt",
    )

    test_df = pd.read_parquet(cfg.dataset_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = test_df[["layout_id", "sample_id", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "steel_true_kg"})
    pred_df["steel_pred_kg"] = test_stat["y_pred"]
    pred_df.to_parquet(out_dir / "predictions_test.parquet", index=False)

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "target_log_mean": float(y_mean),
        "target_log_std": float(y_std),
        "val": _without_arrays(val_stat),
        "test": _without_arrays(test_stat),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def _build_model(
    cfg: GNNSteelTrainConfig, pre, node_dim: int, edge_dim: int, graph_feat_dim: int
) -> LayoutParamGNN:
    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    return LayoutParamGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_feat_dim=graph_feat_dim,
        param_num_dim=len(NUM_COLS),
        param_cat_cardinalities=cat_cardinalities,
        param_emb_dims=auto_param_emb_dims(cat_cardinalities),
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
        conv_type=cfg.conv_type,
    )


def _target_stats(loader) -> tuple[float, float]:
    vals = []
    for batch in loader:
        vals.append(batch.y.detach().cpu().numpy())
    y = np.log1p(np.concatenate(vals, axis=0).astype(np.float64))
    return float(y.mean()), float(max(y.std(ddof=0), 1.0e-6))


def _train_one_epoch(
    model, loader, criterion, optimizer, device: torch.device, y_mean: float, y_std: float
) -> float:
    model.train()
    losses = []
    total_batches = len(loader)
    for i, batch in enumerate(loader):
        batch = batch.to(device)
        y = _normalize_target(batch.y, y_mean, y_std)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        if (i + 1) % 50 == 0 or (i + 1) == total_batches:
            print(f"\rbatch: {i + 1}/{total_batches}, Loss: {losses[-1]:.4f}", flush=True, end="")
    return float(np.mean(losses)) if losses else 0.0


def _eval_regression(model, loader, device: torch.device, y_mean: float, y_std: float) -> dict[str, object]:
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


def _without_arrays(stat: dict[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in stat.items() if k not in {"y_true", "y_pred"}}


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _needs_graph_cache_rebuild(cfg: GNNSteelTrainConfig) -> bool:
    cache_dir = Path(cfg.graph_cache_dir)
    if not cache_dir.exists():
        return True

    summary_path = cache_dir / "summary.json"
    if not summary_path.exists():
        return cfg.graph_repr != "member"

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True

    if summary.get("graph_representation", "member") != cfg.graph_repr:
        return True
    if cfg.graph_repr == "room" and summary.get("feature_version") != ROOM_GRAPH_FEATURE_VERSION:
        return True
    return False
