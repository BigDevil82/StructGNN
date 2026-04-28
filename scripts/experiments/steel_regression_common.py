from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "material_steel_kg"


def target_stats(loader) -> tuple[float, float]:
    vals = []
    for batch in loader:
        vals.append(batch.y.detach().cpu().numpy())
    y = np.log1p(np.concatenate(vals, axis=0).astype(np.float64))
    return float(y.mean()), float(max(y.std(ddof=0), 1.0e-6))


def train_regression(
    *,
    model,
    loaders,
    cfg,
    label: str,
    checkpoint_payload: dict[str, Any],
    checkpoint_name: str,
) -> dict[str, object]:
    y_mean, y_std = target_stats(loaders["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(model, loaders["train"], criterion, optimizer, device, y_mean, y_std)
        val = eval_regression(model, loaders["val"], device, y_mean, y_std)
        score = float(val["mae"])

        if epoch == 0 or (epoch + 1) % max(1, cfg.log_interval) == 0:
            print(
                f"[steel-reg][{label}] epoch={epoch + 1}/{cfg.max_epochs} "
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
                    f"[steel-reg][{label}] early_stop epoch={epoch + 1} "
                    f"best_epoch={best_epoch + 1} best_mae={best_score:.3f}"
                )
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced")

    model.load_state_dict(best_state)
    val = eval_regression(model, loaders["val"], device, y_mean, y_std)
    test = eval_regression(model, loaders["test"], device, y_mean, y_std)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **checkpoint_payload,
            "state_dict": model.cpu().state_dict(),
            "target": TARGET,
            "target_transform": "standardized_log1p",
            "target_log_mean": float(y_mean),
            "target_log_std": float(y_std),
        },
        out_dir / checkpoint_name,
    )

    save_test_predictions(cfg.dataset_path, test, out_dir / "predictions_test.parquet")

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "target_log_mean": float(y_mean),
        "target_log_std": float(y_std),
        "val": drop_arrays(val),
        "test": drop_arrays(test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, y_mean: float, y_std: float) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        y = normalize_target(batch.y, y_mean, y_std)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def eval_regression(model, loader, device: torch.device, y_mean: float, y_std: float) -> dict[str, object]:
    model.eval()
    ys = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            pred = denormalize_target(pred_norm, y_mean, y_std)
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


def save_test_predictions(dataset_path: str, stat: dict[str, object], output_path: str | Path) -> None:
    test_df = pd.read_parquet(dataset_path)
    test_df = test_df[test_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = test_df[["layout_id", "sample_id", TARGET]].copy()
    pred_df = pred_df.rename(columns={TARGET: "steel_true_kg"})
    pred_df["steel_pred_kg"] = stat["y_pred"]
    pred_df.to_parquet(output_path, index=False)


def normalize_target(y: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return (torch.log1p(y) - y_mean) / y_std


def denormalize_target(y: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return torch.expm1(y * y_std + y_mean)


def drop_arrays(stat: dict[str, object]) -> dict[str, float]:
    return {k: float(v) for k, v in stat.items() if k not in {"y_true", "y_pred"}}


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
