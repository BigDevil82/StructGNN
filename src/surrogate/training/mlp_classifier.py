from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .lightgbm_baseline import CLASS_TASK, LAYOUT_FEATURES, PARAM_FEATURES


@dataclass(frozen=True)
class MLPClassifierConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    output_dir: str = r"data\parametric\surrogate_dataset\baseline_mlp_v1"
    seed: int = 42
    batch_size: int = 4096
    max_epochs: int = 50
    early_stop_rounds: int = 10
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dims: tuple[int, int, int] = (256, 128, 64)
    dropout: float = 0.2
    num_workers: int = 0


class BinaryMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple[int, int, int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def run_mlp_classifier(cfg: MLPClassifierConfig) -> dict[str, object]:
    _set_seed(cfg.seed)

    df = pd.read_parquet(cfg.dataset_path)
    required = ["split", CLASS_TASK] + PARAM_FEATURES + LAYOUT_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test split is empty.")

    x_train, x_val, x_test, preprocess = _prepare_features(train_df, val_df, test_df)
    y_train = train_df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)
    y_val = val_df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)
    y_test = test_df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = BinaryMLP(in_dim=x_train.shape[1], hidden_dims=cfg.hidden_dims, dropout=cfg.dropout).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device).squeeze())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_auc = -1.0
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_prob, val_true = _predict_proba(model, val_loader, device)
        val_auc = roc_auc_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                break

    if best_state is None:
        raise RuntimeError("MLP training failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)

    val_prob, val_true = _predict_proba(model, val_loader, device)
    threshold = _find_best_f1_threshold(val_true, val_prob)

    test_prob, test_true = _predict_proba(model, test_loader, device)
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "dataset_path": cfg.dataset_path,
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "test_samples": int(len(test_df)),
        "feature_count": int(x_train.shape[1]),
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val_auc),
        "classification": {
            "roc_auc": float(roc_auc_score(test_true, test_prob)),
            "pr_auc": float(average_precision_score(test_true, test_prob)),
            "f1": float(f1_score(test_true, test_pred, zero_division=0)),
            "precision": float(precision_score(test_true, test_pred, zero_division=0)),
            "recall": float(recall_score(test_true, test_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(test_true, test_pred)),
            "brier": float(np.mean((test_prob - test_true) ** 2)),
            "threshold": float(threshold),
        },
        "config": asdict(cfg),
    }

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train.shape[1]),
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
        },
        out_dir / "clf_final_pass_mlp.pt",
    )

    (out_dir / "preprocess.json").write_text(
        json.dumps(preprocess, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "layout_id": test_df["layout_id"].to_numpy(),
            "sample_id": test_df["sample_id"].to_numpy(),
            "final_pass_true": test_true,
            "final_pass_prob": test_prob,
            "final_pass_pred": test_pred,
        }
    )
    pred_df.to_parquet(out_dir / "test_predictions.parquet", index=False)

    return metrics


def _prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    feature_cols = PARAM_FEATURES + LAYOUT_FEATURES
    cat_cols = ["conc_bot", "site_class", "seismic_group", "intensity"]

    x_train = train_df[feature_cols].copy()
    x_val = val_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()

    for col in cat_cols:
        x_train[col] = x_train[col].astype(str).fillna("<unk>")
        x_val[col] = x_val[col].astype(str).fillna("<unk>")
        x_test[col] = x_test[col].astype(str).fillna("<unk>")

    x_train = pd.get_dummies(x_train, columns=cat_cols, dtype=float)
    x_val = pd.get_dummies(x_val, columns=cat_cols, dtype=float).reindex(
        columns=x_train.columns, fill_value=0.0
    )
    x_test = pd.get_dummies(x_test, columns=cat_cols, dtype=float).reindex(
        columns=x_train.columns, fill_value=0.0
    )

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0, ddof=0)
    train_std = train_std.replace(0.0, 1.0)

    x_train = x_train.fillna(train_mean)
    x_val = x_val.fillna(train_mean)
    x_test = x_test.fillna(train_mean)

    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    x_train_np = x_train.to_numpy(dtype=np.float32)
    x_val_np = x_val.to_numpy(dtype=np.float32)
    x_test_np = x_test.to_numpy(dtype=np.float32)

    preprocess = {
        "feature_columns": list(x_train.columns),
        "mean": [float(v) for v in train_mean.values],
        "std": [float(v) for v in train_std.values],
        "cat_cols": cat_cols,
    }
    return x_train_np, x_val_np, x_test_np, preprocess


def _predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(prob)
            ys.append(yb.numpy())

    y_prob = np.concatenate(probs, axis=0).astype(np.float64)
    y_true = np.concatenate(ys, axis=0).astype(np.int64)
    return y_prob, y_true


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
