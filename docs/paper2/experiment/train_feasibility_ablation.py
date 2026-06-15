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
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.gnn.dataset import CAT_COLS, NUM_COLS, GNNDataConfig, build_dataloaders
from src.surrogate.gnn.model import ParamEncoder, auto_param_emb_dims
from src.surrogate.training.lightgbm_baseline import CLASS_TASK


@dataclass(frozen=True)
class FeasibilityAblationConfig:
    baseline: str
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    graph_repr: str = "room"
    graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    output_dir: str = r"data\parametric\ckpt\feasibility_ablation"
    seed: int = 42
    batch_size: int = 512
    max_epochs: int = 8
    early_stop_rounds: int = 2
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    gnn_layers: int = 3
    conv_type: str = "sage"
    dropout: float = 0.1
    num_workers: int = 0
    log_interval: int = 1
    screening_target_recall: float = 0.995


class ParamOnlyFeasibilityMLP(nn.Module):
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


class ParamGraphFeatFeasibilityMLP(nn.Module):
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


def run(cfg: FeasibilityAblationConfig) -> dict[str, object]:
    _set_seed(cfg.seed)
    loaders, pre, (_, _, graph_feat_dim) = build_dataloaders(
        GNNDataConfig(
            dataset_path=cfg.dataset_path,
            graph_cache_dir=cfg.graph_cache_dir,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        ),
        target_col=CLASS_TASK,
    )

    cat_cardinalities = [len(pre.cat_vocab[c]) for c in CAT_COLS]
    if cfg.baseline == "param_only":
        model = ParamOnlyFeasibilityMLP(len(NUM_COLS), cat_cardinalities, cfg.hidden_dim, cfg.dropout)
    elif cfg.baseline == "param_graph_feat":
        model = ParamGraphFeatFeasibilityMLP(
            graph_feat_dim, len(NUM_COLS), cat_cardinalities, cfg.hidden_dim, cfg.dropout
        )
    else:
        raise ValueError(f"Unknown baseline: {cfg.baseline}")

    return train_binary(
        model=model,
        loaders=loaders,
        cfg=cfg,
        label=cfg.baseline,
        checkpoint_payload={
            "baseline": cfg.baseline,
            "preprocess": pre.export(),
            "cat_cardinalities": cat_cardinalities,
            "graph_feat_dim": int(graph_feat_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
        },
    )


def train_binary(
    *,
    model: nn.Module,
    loaders,
    cfg: FeasibilityAblationConfig,
    label: str,
    checkpoint_payload: dict[str, object],
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(loaders["train"], device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_score = -1.0
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val = eval_binary(model, loaders["val"], device)
        score = float(val["pr_auc"])

        if epoch == 0 or (epoch + 1) % max(1, cfg.log_interval) == 0:
            threshold = find_best_f1_threshold(val["y_true"], val["y_prob"])
            y_hat = (val["y_prob"] >= threshold).astype(int)
            print(
                f"[feas-ablation][{label}] epoch={epoch + 1}/{cfg.max_epochs} "
                f"train_loss={train_loss:.6f} val_pr_auc={val['pr_auc']:.6f} "
                f"val_f1={f1_score(val['y_true'], y_hat, zero_division=0):.6f} "
                f"best_pr_auc={best_score:.6f}"
            )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                print(
                    f"[feas-ablation][{label}] early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch + 1} best_pr_auc={best_score:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")

    model.load_state_dict(best_state)
    val = eval_binary(model, loaders["val"], device)
    test = eval_binary(model, loaders["test"], device)
    threshold = find_best_f1_threshold(val["y_true"], val["y_prob"])
    screen_threshold, val_screen = find_screening_threshold(
        val["y_true"], val["y_prob"], cfg.screening_target_recall
    )
    test_pred = (test["y_prob"] >= threshold).astype(int)
    test_screen = screening_metrics(test["y_true"], test["y_prob"], screen_threshold)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint_payload,
            "state_dict": model.cpu().state_dict(),
            "target": CLASS_TASK,
            "threshold": float(threshold),
            "screening_threshold": float(screen_threshold),
            "screening_target_recall": float(cfg.screening_target_recall),
        },
        out_dir / "feasibility_ablation.pt",
    )
    save_test_predictions(cfg.dataset_path, test, test_pred, out_dir / "predictions_test.parquet")

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "threshold": float(threshold),
        "screening_threshold": float(screen_threshold),
        "val": binary_metrics(val["y_true"], val["y_prob"], (val["y_prob"] >= threshold).astype(int)),
        "test": binary_metrics(test["y_true"], test["y_prob"], test_pred),
        "screening": {
            "target_recall": float(cfg.screening_target_recall),
            "val": val_screen,
            "test": test_screen,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
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


def eval_binary(model, loader, device: torch.device) -> dict[str, object]:
    model.eval()
    ys = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prob = torch.sigmoid(model(batch))
            ys.append(batch.y.detach().cpu().numpy())
            probs.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0).astype(np.int64)
    y_prob = np.concatenate(probs, axis=0).astype(np.float64)
    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "pr_auc": safe_auc(average_precision_score, y_true, y_prob),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
    }


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "samples": float(len(y_true)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_auc(average_precision_score, y_true, y_prob),
    }


def find_best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr


def find_screening_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float,
) -> tuple[float, dict[str, float]]:
    best_thr = 0.0
    best = screening_metrics(y_true, y_prob, best_thr)
    for thr in np.arange(0.0, 1.001, 0.002):
        metrics = screening_metrics(y_true, y_prob, float(thr))
        if metrics["feasible_recall"] >= target_recall and metrics["reject_rate"] > best["reject_rate"]:
            best_thr = float(thr)
            best = metrics
    return best_thr, best


def screening_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    feasible = y_true.astype(bool)
    reject = y_prob < threshold
    keep = ~reject
    false_reject = reject & feasible
    true_reject = reject & ~feasible
    kept_feasible = keep & feasible

    return {
        "threshold": float(threshold),
        "reject_rate": float(reject.mean()),
        "keep_rate": float(keep.mean()),
        "feasible_recall": float(kept_feasible.sum() / max(int(feasible.sum()), 1)),
        "false_reject_rate": float(false_reject.sum() / max(int(feasible.sum()), 1)),
        "false_reject_count": float(false_reject.sum()),
        "reject_infeasible_precision": float(true_reject.sum() / max(int(reject.sum()), 1)),
        "kept_feasible_rate": float(kept_feasible.sum() / max(int(keep.sum()), 1)),
    }


def save_test_predictions(
    dataset_path: str, test: dict[str, object], test_pred: np.ndarray, output_path: str | Path
) -> None:
    test_df = pd.read_parquet(dataset_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = test_df[["layout_id", "sample_id", CLASS_TASK]].copy()
    pred_df = pred_df.rename(columns={CLASS_TASK: "final_pass_true"})
    pred_df["pred_final_pass_prob"] = test["y_prob"]
    pred_df["pred_final_pass"] = test_pred
    pred_df.to_parquet(output_path, index=False)


def _pos_weight(loader, device: torch.device) -> torch.Tensor:
    pos = 0.0
    total = 0.0
    for batch in loader:
        y = batch.y.detach().cpu().numpy()
        pos += float(y.sum())
        total += float(len(y))
    neg = max(total - pos, 1.0)
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32, device=device)


def safe_auc(fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(fn(y_true, y_prob))


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Feasibility classification ablations outside production src code.")
    p.add_argument("--baseline", choices=("param_only", "param_graph_feat"), required=True)
    p.add_argument("--dataset-path", default=FeasibilityAblationConfig.dataset_path)
    p.add_argument("--graph-repr", choices=("member", "room"), default="room")
    p.add_argument("--graph-cache-dir", default=FeasibilityAblationConfig.graph_cache_dir)
    p.add_argument("--output-dir", default=FeasibilityAblationConfig.output_dir)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=8)
    p.add_argument("--early-stop-rounds", type=int, default=2)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gnn-layers", type=int, default=3)
    p.add_argument("--conv-type", choices=("sage", "gine"), default="sage")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--screening-target-recall", type=float, default=0.995)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = FeasibilityAblationConfig(**vars(args))
    metrics = run(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
