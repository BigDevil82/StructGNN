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

CAT_COLS = ["conc_bot", "site_class", "seismic_group", "intensity"]


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        num_dim: int,
        cat_cardinalities: list[int],
        emb_dims: list[int],
        hidden_dims: tuple[int, int, int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_bn = nn.BatchNorm1d(num_dim)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, emb_dim) for cardinality, emb_dim in zip(cat_cardinalities, emb_dims)]
        )

        in_dim = num_dim + int(sum(emb_dims))
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
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x_num = self.num_bn(x_num)
        emb = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1)
        return self.mlp(x).squeeze(1)


@dataclass(frozen=True)
class MLPEmbeddingKFoldConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    fold_map_path: str = r"data\parametric\surrogate_dataset\splits\layout_group_folds.parquet"
    output_dir: str = r"data\parametric\surrogate_dataset\baseline_mlp_embedding_kfold"
    seed: int = 42
    batch_size: int = 4096
    max_epochs: int = 50
    early_stop_rounds: int = 10
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    hidden_dims: tuple[int, int, int] = (256, 128, 64)
    dropout: float = 0.2
    num_workers: int = 0
    log_interval: int = 5


def run_mlp_embedding_kfold(cfg: MLPEmbeddingKFoldConfig) -> dict[str, object]:
    _set_seed(cfg.seed)

    df = pd.read_parquet(cfg.dataset_path)
    fold_map = pd.read_parquet(cfg.fold_map_path)

    if "layout_id" not in fold_map.columns or "group_fold" not in fold_map.columns:
        raise ValueError("Fold map must contain layout_id and group_fold.")

    required = ["layout_id", CLASS_TASK] + PARAM_FEATURES + LAYOUT_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df = df.drop(columns=["group_fold"], errors="ignore").merge(
        fold_map[["layout_id", "group_fold"]], on="layout_id", how="left", validate="many_to_one"
    )
    if df["group_fold"].isna().any():
        raise ValueError("Some samples have no group_fold assignment.")
    df["group_fold"] = df["group_fold"].astype(int)

    if "split" in df.columns and df["split"].isin(["train", "val"]).any():
        train_pool = df[df["split"].isin(["train", "val"])].copy()
        train_pool_tag = "split=train+val"
    else:
        train_pool = df
        train_pool_tag = "all_samples"

    if train_pool.empty:
        raise ValueError("No training samples found for K-Fold. Please check split assignments.")

    fold_ids = sorted(train_pool["group_fold"].unique().tolist())
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_summaries: list[dict[str, object]] = []
    oof_parts: list[pd.DataFrame] = []
    fold_artifacts: list[dict[str, object]] = []

    for fold_idx, fold_id in enumerate(fold_ids, start=1):
        train_df = train_pool[train_pool["group_fold"] != fold_id].copy()
        val_df = train_pool[train_pool["group_fold"] == fold_id].copy()
        if train_df.empty or val_df.empty:
            continue

        print(
            f"[surrogate][mlp-embedding-kfold] fold {fold_idx}/{len(fold_ids)} "
            f"train={len(train_df)} val={len(val_df)}"
        )

        (
            x_train_num,
            x_train_cat,
            x_val_num,
            x_val_cat,
            preprocess,
        ) = _prepare_fold_features(train_df, val_df)

        y_train = train_df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)
        y_val = val_df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)

        model, best_epoch, best_val_auc = _train_one_fold(
            cfg,
            x_train_num,
            x_train_cat,
            y_train,
            x_val_num,
            x_val_cat,
            y_val,
            fold_id=fold_id,
        )

        val_prob = predict_prob(
            model,
            x_val_num,
            x_val_cat,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        threshold = _find_best_f1_threshold(y_val.astype(int), val_prob)
        val_pred = (val_prob >= threshold).astype(int)
        val_metrics = _classification_metrics(y_val.astype(int), val_prob, val_pred)

        fold_dir = out_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model_info = {
            "num_dim": int(x_train_num.shape[1]),
            "cat_cardinalities": [len(preprocess["cat_vocab"][col]) for col in CAT_COLS],
            "emb_dims": _embedding_dims([len(preprocess["cat_vocab"][col]) for col in CAT_COLS]),
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": float(cfg.dropout),
        }

        torch.save(
            {
                "state_dict": model.state_dict(),
                **model_info,
            },
            fold_dir / "clf_final_pass_mlp_embedding.pt",
        )
        (fold_dir / "preprocess.json").write_text(
            json.dumps(preprocess, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        fold_summary = {
            "fold": int(fold_id),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "best_epoch": int(best_epoch),
            "best_val_roc_auc": float(best_val_auc),
            "threshold": float(threshold),
            "metrics": val_metrics,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(fold_summary, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        fold_summaries.append(fold_summary)

        oof = val_df[["layout_id", "sample_id", CLASS_TASK]].copy()
        oof["group_fold"] = fold_id
        oof["final_pass_prob"] = val_prob
        oof["final_pass_pred"] = val_pred
        oof_parts.append(oof)

        fold_artifacts.append(
            {
                "fold_id": int(fold_id),
                "model": model,
                "preprocess": preprocess,
                "threshold": float(threshold),
            }
        )

    if not oof_parts:
        raise RuntimeError("No folds were trained. Please check fold assignments.")

    oof_df = pd.concat(oof_parts, ignore_index=True)
    oof_df = oof_df.rename(columns={CLASS_TASK: "final_pass_true"})
    oof_path = out_dir / "oof_predictions.parquet"
    oof_df.to_parquet(oof_path, index=False)

    oof_metrics = _classification_metrics(
        oof_df["final_pass_true"].astype(int).to_numpy(),
        oof_df["final_pass_prob"].astype(float).to_numpy(),
        oof_df["final_pass_pred"].astype(int).to_numpy(),
    )

    ensemble_test_summary: dict[str, object] | None = None
    if "split" in df.columns:
        test_df = df[df["split"] == "test"].copy()
        if not test_df.empty:
            ensemble_probs = []
            thresholds = []
            for artifact in fold_artifacts:
                probs = _predict_with_artifact(test_df, artifact, cfg.batch_size, cfg.num_workers)
                ensemble_probs.append(probs)
                thresholds.append(float(artifact["threshold"]))

            mean_prob = np.mean(np.stack(ensemble_probs, axis=0), axis=0)
            thr = float(np.mean(thresholds))
            pred = (mean_prob >= thr).astype(int)

            test_true = test_df[CLASS_TASK].astype(int).to_numpy()
            test_metrics = _classification_metrics(test_true, mean_prob, pred)

            ens_df = test_df[["layout_id", "sample_id", CLASS_TASK]].copy()
            ens_df = ens_df.rename(columns={CLASS_TASK: "final_pass_true"})
            ens_df["final_pass_prob"] = mean_prob
            ens_df["final_pass_pred"] = pred
            ens_df.to_parquet(out_dir / "ensemble_test_predictions.parquet", index=False)

            ensemble_test_summary = {
                "test_samples": int(len(test_df)),
                "threshold": thr,
                "metrics": test_metrics,
            }

    summary = {
        "config": asdict(cfg),
        "train_pool": train_pool_tag,
        "num_folds": int(len(fold_summaries)),
        "folds": fold_summaries,
        "oof_metrics": oof_metrics,
        "ensemble_test": ensemble_test_summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return summary


def _train_one_fold(
    cfg: MLPEmbeddingKFoldConfig,
    x_train_num: np.ndarray,
    x_train_cat: np.ndarray,
    y_train: np.ndarray,
    x_val_num: np.ndarray,
    x_val_cat: np.ndarray,
    y_val: np.ndarray,
    fold_id: int,
) -> tuple[EmbeddingMLP, int, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_num), torch.from_numpy(x_train_cat), torch.from_numpy(y_train)
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val_num), torch.from_numpy(x_val_cat), torch.from_numpy(y_val)),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    cat_cardinalities = [int(x_train_cat[:, i].max()) + 1 for i in range(x_train_cat.shape[1])]
    emb_dims = _embedding_dims(cat_cardinalities)

    model = EmbeddingMLP(
        num_dim=x_train_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        emb_dims=emb_dims,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device).squeeze())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val_pr_auc = -1.0
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb_num, xb_cat)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_prob, val_true = _predict_proba_from_loader(model, val_loader, device)
        val_pr_auc = average_precision_score(val_true, val_prob) if len(np.unique(val_true)) > 1 else 0.5

        if epoch == 0 or (epoch + 1) % max(cfg.log_interval, 1) == 0:
            print(
                f"[surrogate][mlp-embedding-kfold] fold={fold_id + 1} "
                f"epoch={epoch + 1}/{cfg.max_epochs} val_pr_auc={val_pr_auc:.6f} best={best_val_pr_auc:.6f}"
            )

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = float(val_pr_auc)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= cfg.early_stop_rounds:
                print(
                    f"[surrogate][mlp-embedding-kfold] fold={fold_id + 1} early_stop "
                    f"epoch={epoch + 1} best_epoch={best_epoch + 1} best_pr_auc={best_val_pr_auc:.6f}"
                )
                break

    if best_state is None:
        raise RuntimeError("KFold model failed to produce a valid checkpoint.")

    model.load_state_dict(best_state)
    return model.cpu(), int(best_epoch), float(best_val_pr_auc)


def _prepare_fold_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    all_cols = PARAM_FEATURES + LAYOUT_FEATURES
    num_cols = [c for c in all_cols if c not in CAT_COLS]

    x_train_num = train_df[num_cols].copy()
    x_val_num = val_df[num_cols].copy()

    train_mean = x_train_num.mean(axis=0)
    train_std = x_train_num.std(axis=0, ddof=0).replace(0.0, 1.0)

    x_train_num = x_train_num.fillna(train_mean)
    x_val_num = x_val_num.fillna(train_mean)

    x_train_num = ((x_train_num - train_mean) / train_std).to_numpy(dtype=np.float32)
    x_val_num = ((x_val_num - train_mean) / train_std).to_numpy(dtype=np.float32)

    cat_vocab: dict[str, dict[str, int]] = {}
    x_train_cat_cols: list[np.ndarray] = []
    x_val_cat_cols: list[np.ndarray] = []

    for col in CAT_COLS:
        tr = train_df[col].astype(str).fillna("<unk>")
        va = val_df[col].astype(str).fillna("<unk>")

        unique_vals = sorted(v for v in tr.unique() if v != "<unk>")
        vocab = {"<unk>": 0}
        for i, val in enumerate(unique_vals, start=1):
            vocab[val] = i
        cat_vocab[col] = vocab

        x_train_cat_cols.append(tr.map(lambda v: vocab.get(v, 0)).to_numpy(dtype=np.int64))
        x_val_cat_cols.append(va.map(lambda v: vocab.get(v, 0)).to_numpy(dtype=np.int64))

    x_train_cat = np.stack(x_train_cat_cols, axis=1)
    x_val_cat = np.stack(x_val_cat_cols, axis=1)

    preprocess = {
        "num_cols": num_cols,
        "cat_cols": CAT_COLS,
        "cat_vocab": cat_vocab,
        "num_mean": [float(v) for v in train_mean.values],
        "num_std": [float(v) for v in train_std.values],
    }

    return x_train_num, x_train_cat, x_val_num, x_val_cat, preprocess


def transform_with_preprocess(
    df: pd.DataFrame, preprocess: dict[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    num_cols: list[str] = list(preprocess["num_cols"])
    cat_cols: list[str] = list(preprocess.get("cat_cols", CAT_COLS))
    cat_vocab: dict[str, dict[str, int]] = dict(preprocess["cat_vocab"])

    x_num = df[num_cols].copy()
    num_mean = pd.Series(preprocess["num_mean"], index=num_cols, dtype=float)
    num_std = pd.Series(preprocess["num_std"], index=num_cols, dtype=float)
    x_num = x_num.fillna(num_mean)
    x_num = ((x_num - num_mean) / num_std).to_numpy(dtype=np.float32)

    cat_cols_np = []
    for col in cat_cols:
        vocab = cat_vocab[col]
        raw = df[col].astype(str).fillna("<unk>")
        cat_cols_np.append(raw.map(lambda v: vocab.get(v, 0)).to_numpy(dtype=np.int64))
    x_cat = np.stack(cat_cols_np, axis=1)
    return x_num, x_cat


def _predict_with_artifact(
    df: pd.DataFrame,
    artifact: dict[str, object],
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    preprocess = artifact["preprocess"]
    x_num, x_cat = transform_with_preprocess(df, preprocess)

    cat_cardinalities = [len(preprocess["cat_vocab"][col]) for col in CAT_COLS]
    emb_dims = _embedding_dims(cat_cardinalities)

    model = EmbeddingMLP(
        num_dim=x_num.shape[1],
        cat_cardinalities=cat_cardinalities,
        emb_dims=emb_dims,
        hidden_dims=(256, 128, 64),
        dropout=0.2,
    )
    model.load_state_dict(artifact["model"].state_dict())

    return predict_prob(model, x_num, x_cat, batch_size=batch_size, num_workers=num_workers)


def predict_prob(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_num), torch.from_numpy(x_cat)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for xb_num, xb_cat in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            logits = model(xb_num, xb_cat)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(prob)
    return np.concatenate(probs, axis=0).astype(np.float64)


def _predict_proba_from_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for xb_num, xb_cat, yb in loader:
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device)
            logits = model(xb_num, xb_cat)
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            probs.append(prob)
            ys.append(yb.numpy())
    y_prob = np.concatenate(probs, axis=0).astype(np.float64)
    y_true = np.concatenate(ys, axis=0).astype(np.int64)
    return y_prob, y_true


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }
    return metrics


def _embedding_dims(cardinalities: list[int]) -> list[int]:
    return [min(32, max(4, (card + 1) // 2)) for card in cardinalities]


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
