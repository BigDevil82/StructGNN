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


def target_raw_stats(loader) -> dict[str, float]:
    vals = []
    for batch in loader:
        vals.append(batch.y.detach().cpu().numpy())
    y = np.concatenate(vals, axis=0).astype(np.float64)
    return {
        "median": float(np.median(y)),
        "q80": float(np.quantile(y, 0.8)),
    }


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
    y_raw = target_raw_stats(loaders["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_score = float("inf")
    best_epoch = -1
    bad_rounds = 0

    for epoch in range(cfg.max_epochs):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            y_mean,
            y_std,
            cfg=cfg,
            y_raw=y_raw,
        )
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
    uncalibrated_val = val
    uncalibrated_test = test
    calibration = fit_calibrator(val, getattr(cfg, "calibration", "none"))
    if calibration["type"] != "none":
        val = apply_calibrator(val, calibration)
        test = apply_calibrator(test, calibration)

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
            "target_median_kg": y_raw["median"],
            "target_q80_kg": y_raw["q80"],
            "calibration": calibration,
        },
        out_dir / checkpoint_name,
    )

    save_test_predictions(cfg.dataset_path, test, out_dir / "predictions_test.parquet")

    metrics = {
        "config": asdict(cfg),
        "best_epoch": int(best_epoch),
        "target_log_mean": float(y_mean),
        "target_log_std": float(y_std),
        "target_median_kg": y_raw["median"],
        "target_q80_kg": y_raw["q80"],
        "calibration": calibration,
        "uncalibrated_val": drop_arrays(uncalibrated_val),
        "uncalibrated_test": drop_arrays(uncalibrated_test),
        "val": drop_arrays(val),
        "test": drop_arrays(test),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    y_mean: float,
    y_std: float,
    *,
    cfg,
    y_raw: dict[str, float],
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = batch.to(device)
        y = normalize_target(batch.y, y_mean, y_std)
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = loss_fn(pred, y, batch.y, cfg=cfg, y_mean=y_mean, y_std=y_std, y_raw=y_raw)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


def loss_fn(
    pred_norm: torch.Tensor,
    y_norm: torch.Tensor,
    y_kg: torch.Tensor,
    *,
    cfg,
    y_mean: float,
    y_std: float,
    y_raw: dict[str, float],
) -> torch.Tensor:
    loss_type = getattr(cfg, "loss_type", "standard")
    base = torch.nn.functional.smooth_l1_loss(pred_norm, y_norm, beta=0.5, reduction="none")

    if loss_type == "standard":
        return base.mean()
    if loss_type == "kg_aux":
        pred_kg = denormalize_target(pred_norm, y_mean, y_std)
        scale = float(getattr(cfg, "kg_scale", 200000.0))
        alpha = float(getattr(cfg, "kg_alpha", 0.2))
        kg_loss = torch.nn.functional.smooth_l1_loss(pred_kg / scale, y_kg / scale, beta=0.5, reduction="none")
        return base.mean() + alpha * kg_loss.mean()
    if loss_type == "q80_weight":
        high_weight = float(getattr(cfg, "high_weight", 1.0))
        weight = 1.0 + high_weight * (y_kg > float(y_raw["q80"])).float()
        return (weight * base).mean()
    if loss_type == "continuous_weight":
        median = max(float(y_raw["median"]), 1.0)
        weight = (y_kg / median).clamp(0.5, 3.0)
        return (weight * base).mean()
    raise ValueError(f"Unknown loss_type: {loss_type}")


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


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    y_true = y_true.astype(np.float64)
    y_pred = np.maximum(y_pred.astype(np.float64), 0.0)
    err = y_pred - y_true
    return {
        "samples": float(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs(err) / np.maximum(y_true, 1.0))),
        "bias": float(err.mean()),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit_calibrator(stat: dict[str, object], mode: str) -> dict[str, object]:
    if mode == "none":
        return {"type": "none"}
    y_true = stat["y_true"].astype(np.float64)
    y_pred = stat["y_pred"].astype(np.float64)
    if mode == "linear":
        a, b = np.polyfit(y_pred, y_true, deg=1)
        return {"type": "linear", "a": float(a), "b": float(b)}
    if mode == "quantile_bias":
        qs = np.quantile(y_pred, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        qs[0] = -np.inf
        qs[-1] = np.inf
        bins = np.digitize(y_pred, qs[1:-1], right=False)
        bias = [float((y_true[bins == i] - y_pred[bins == i]).mean()) for i in range(3)]
        return {"type": "quantile_bias", "edges": [float(v) for v in qs[1:-1]], "bias": bias}
    raise ValueError(f"Unknown calibration: {mode}")


def apply_calibrator(stat: dict[str, object], calibration: dict[str, object]) -> dict[str, object]:
    y_true = stat["y_true"]
    y_pred = stat["y_pred"]
    if calibration["type"] == "linear":
        y_pred = float(calibration["a"]) * y_pred + float(calibration["b"])
    elif calibration["type"] == "quantile_bias":
        edges = np.array(calibration["edges"], dtype=np.float64)
        bias = np.array(calibration["bias"], dtype=np.float64)
        bins = np.digitize(y_pred, edges, right=False)
        y_pred = y_pred + bias[bins]
    elif calibration["type"] != "none":
        raise ValueError(f"Unknown calibration type: {calibration['type']}")
    return metrics_from_arrays(y_true, y_pred)


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
