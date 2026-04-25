from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.surrogate.gnn.dataset import CAT_COLS, NUM_COLS, GNNDataConfig, build_dataloaders
from src.surrogate.gnn.graph_data import LayoutGraphCacheConfig, build_layout_graph_cache
from src.surrogate.gnn.model import LayoutParamGNN, auto_param_emb_dims
from src.surrogate.training.lightgbm_baseline import CLASS_TASK


@dataclass(frozen=True)
class GNNTrainConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    layout_json_dir: str = r"data\dxf\cad_json_data\fem_raw"
    layout_dxf_dir: str = r"data\dxf\fem_raw"
    graph_cache_dir: str = r"data\parametric\surrogate_dataset\gnn_graph_cache"
    graph_repr: str = "member"
    output_dir: str = r"data\parametric\surrogate_dataset\baseline_gnn_final_pass"
    seed: int = 42
    batch_size: int = 128
    max_epochs: int = 50
    early_stop_rounds: int = 10
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    gnn_layers: int = 3
    dropout: float = 0.2
    num_workers: int = 0
    log_interval: int = 1
    rebuild_graph_cache: bool = False
    merge_members: bool = True


def run_gnn_train(cfg: GNNTrainConfig) -> dict[str, object]:
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
        )
    )

    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    emb_dims = auto_param_emb_dims(cat_cardinalities)

    model = LayoutParamGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        graph_feat_dim=graph_feat_dim,
        param_num_dim=len(NUM_COLS),
        param_cat_cardinalities=cat_cardinalities,
        param_emb_dims=emb_dims,
        hidden_dim=cfg.hidden_dim,
        gnn_layers=cfg.gnn_layers,
        dropout=cfg.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pos_weight = _estimate_pos_weight(loaders["train"], device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_pr_auc = -1.0
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        train_loss = _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_stat = _eval_binary(model, loaders["val"], device)
        val_pr_auc = val_stat["pr_auc"]

        if epoch == 0 or (epoch + 1) % max(cfg.log_interval, 1) == 0:
            print(
                f"[surrogate][gnn] epoch={epoch + 1}/{cfg.max_epochs} "
                f"train_loss={train_loss:.6f} val_pr_auc={val_pr_auc:.6f} best={best_pr_auc:.6f}"
            )

        if val_pr_auc > best_pr_auc:
            best_pr_auc = float(val_pr_auc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                print(
                    f"[surrogate][gnn] early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch + 1} best_pr_auc={best_pr_auc:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("GNN failed to produce a valid checkpoint")

    model.load_state_dict(best_state)

    val_stat = _eval_binary(model, loaders["val"], device)
    threshold = _find_best_f1_threshold(val_stat["y_true"], val_stat["y_prob"])

    test_stat = _eval_binary(model, loaders["test"], device)
    test_pred = (test_stat["y_prob"] >= threshold).astype(int)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            },
            "preprocess": pre.export(),
            "threshold": float(threshold),
        },
        out_dir / "gnn_final_pass.pt",
    )

    test_df = pd.read_parquet(cfg.dataset_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = test_df[["layout_id", "sample_id", CLASS_TASK]].copy()
    pred_df = pred_df.rename(columns={CLASS_TASK: "final_pass_true"})
    pred_df["pred_final_pass_prob"] = test_stat["y_prob"]
    pred_df["pred_final_pass"] = test_pred
    pred_df.to_parquet(out_dir / "predictions_test.parquet", index=False)

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "val": _binary_metrics(
            val_stat["y_true"], val_stat["y_prob"], (val_stat["y_prob"] >= threshold).astype(int)
        ),
        "test": _binary_metrics(test_stat["y_true"], test_stat["y_prob"], test_pred),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def _estimate_pos_weight(loader, device: torch.device) -> torch.Tensor:
    pos = 0.0
    total = 0.0
    for batch in loader:
        y = batch.y.detach().cpu().numpy()
        pos += float(y.sum())
        total += float(len(y))
    neg = max(total - pos, 1.0)
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)


def _train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def _eval_binary(model, loader, device: torch.device) -> dict[str, np.ndarray | float]:
    model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            prob = torch.sigmoid(logits)
            ys.append(batch.y.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).astype(np.int64)
    y_prob = np.concatenate(probs, axis=0).astype(np.float64)
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    return {"y_true": y_true, "y_prob": y_prob, "pr_auc": pr_auc}


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "samples": float(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def _find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _needs_graph_cache_rebuild(cfg: GNNTrainConfig) -> bool:
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

    return summary.get("graph_representation", "member") != cfg.graph_repr
