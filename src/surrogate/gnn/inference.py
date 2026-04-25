from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.loader import DataLoader

from src.surrogate.gnn.dataset import ParamPreprocessor, SurrogateGNNDataset
from src.surrogate.gnn.model import LayoutParamGNN


def predict_with_gnn(
    df: pd.DataFrame,
    artifact_path: str | Path,
    graph_cache_dir: str | Path,
    threshold: float | None = None,
    screening_threshold: float | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    report_metrics: bool = True,
) -> pd.DataFrame:
    artifact = torch.load(Path(artifact_path), map_location="cpu", weights_only=True)
    model_cfg = artifact["model_cfg"]
    pre = ParamPreprocessor.from_dict(artifact["preprocess"])

    model = LayoutParamGNN(**model_cfg)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    ds = SurrogateGNNDataset(df, graph_cache_dir=graph_cache_dir, preprocessor=pre)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())

    y_prob = np.concatenate(probs, axis=0).astype(np.float64)
    threshold_val = float(artifact.get("threshold", 0.5)) if threshold is None else float(threshold)
    screening_threshold_val = artifact.get("screening_threshold") if screening_threshold is None else screening_threshold

    pred = pd.DataFrame(index=df.index)
    pred["pred_final_pass_prob"] = y_prob
    pred["pred_final_pass"] = y_prob >= threshold_val
    if screening_threshold_val is not None:
        screening_threshold_val = float(screening_threshold_val)
        pred["screening_threshold"] = screening_threshold_val
        pred["pred_screen_reject"] = y_prob < screening_threshold_val

    if report_metrics and "final_pass" in df.columns:
        y_true = df["final_pass"].astype(int).to_numpy()
        y_pred = pred["pred_final_pass"].astype(int).to_numpy()
        metrics = {
            "samples": float(len(y_true)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        print("[surrogate][gnn] prediction metrics")
        print(json.dumps({"classification": metrics}, ensure_ascii=False, indent=2))

    return pred
