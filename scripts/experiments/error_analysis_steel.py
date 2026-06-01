from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.training.lightgbm_baseline import PARAM_FEATURES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Error analysis for steel usage regression predictions.")
    p.add_argument(
        "--predictions-path",
        default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\predictions_test.parquet",
    )
    p.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    p.add_argument("--graph-cache-dir", default=r"data\parametric\cache\gnn_room_graph_cache")
    p.add_argument(
        "--output-dir",
        default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\error_analysis",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_analysis_frame(args.predictions_path, args.dataset_path, args.graph_cache_dir)
    df["err"] = df["steel_pred_kg"] - df["steel_true_kg"]
    df["abs_err"] = df["err"].abs()
    df["ape"] = df["abs_err"] / df["steel_true_kg"].clip(lower=1.0)

    y_bucket = _bucket_metrics(df, "steel_true_kg", "y_true_bucket")
    y_bucket.to_csv(out_dir / "metrics_by_y_true_bucket.csv", index=False)

    graph_tables = {}
    for col in ["node_count", "edge_count", "bbox_area"]:
        table = _bucket_metrics(df, col, f"{col}_bucket")
        table.to_csv(out_dir / f"metrics_by_{col}_bucket.csv", index=False)
        graph_tables[col] = table

    param_metrics = _param_metrics(df)
    param_metrics.to_csv(out_dir / "metrics_by_param_bucket.csv", index=False)

    layout_metrics = _layout_metrics(df)
    layout_metrics.to_csv(out_dir / "metrics_by_layout.csv", index=False)

    fig_path = out_dir / "param_error_bars.png"
    _plot_param_error_bars(param_metrics, fig_path)

    hist_path = out_dir / "steel_true_hist.png"
    _plot_true_steel_histogram(df, hist_path)
    norm_hist_path = out_dir / "steel_true_hist_normalized.png"
    _plot_normalized_true_histogram(df, norm_hist_path)
    summary = {
        "overall": _metrics(df),
        "outputs": {
            "y_true_bucket": str(out_dir / "metrics_by_y_true_bucket.csv"),
            "node_count_bucket": str(out_dir / "metrics_by_node_count_bucket.csv"),
            "edge_count_bucket": str(out_dir / "metrics_by_edge_count_bucket.csv"),
            "bbox_area_bucket": str(out_dir / "metrics_by_bbox_area_bucket.csv"),
            "param_bucket": str(out_dir / "metrics_by_param_bucket.csv"),
            "layout": str(out_dir / "metrics_by_layout.csv"),
            "param_error_plot": str(fig_path),
            "steel_true_hist": str(hist_path),
            "steel_true_hist_normalized": str(norm_hist_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    report = _make_report(summary, y_bucket, graph_tables, param_metrics, layout_metrics)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


def _load_analysis_frame(predictions_path: str, dataset_path: str, graph_cache_dir: str) -> pd.DataFrame:
    pred = pd.read_parquet(predictions_path)
    meta_cols = ["layout_id", "sample_id", "split", "bbox_area"] + PARAM_FEATURES
    meta = pd.read_parquet(dataset_path, columns=meta_cols)
    meta = meta[meta["split"] == "test"].copy()
    df = pred.merge(meta, on=["layout_id", "sample_id"], how="left", validate="one_to_one")
    if df["split"].isna().any():
        missing = int(df["split"].isna().sum())
        raise ValueError(f"Missing dataset metadata for {missing} prediction rows")

    graph_stats = _load_graph_stats(graph_cache_dir)
    df = df.merge(graph_stats, on="layout_id", how="left", validate="many_to_one")
    if df[["node_count", "edge_count"]].isna().any().any():
        raise ValueError("Missing graph stats for some layouts")
    return df


def _load_graph_stats(graph_cache_dir: str) -> pd.DataFrame:
    rows = []
    for path in Path(graph_cache_dir).glob("*.pt"):
        g = torch.load(path, map_location="cpu", weights_only=True)
        rows.append(
            {
                "layout_id": path.stem,
                "node_count": int(g["x"].shape[0]),
                "edge_count": int(g["edge_index"].shape[1]),
            }
        )
    return pd.DataFrame(rows)


def _bucket_metrics(df: pd.DataFrame, col: str, bucket_col: str) -> pd.DataFrame:
    work = df.copy()
    work[bucket_col] = _qbucket(work[col])
    rows = []
    for bucket, part in work.groupby(bucket_col, observed=True):
        lo = float(part[col].min())
        hi = float(part[col].max())
        rows.append(
            {
                "bucket": str(bucket),
                "range_min": lo,
                "range_max": hi,
                **_metrics(part),
            }
        )
    return pd.DataFrame(rows)


def _param_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in PARAM_FEATURES:
        labels = _value_or_quantile_bucket(df[col])
        for bucket, part in df.assign(_bucket=labels).groupby("_bucket", observed=True):
            rows.append(
                {
                    "param": col,
                    "bucket": str(bucket),
                    **_metrics(part),
                }
            )
    return pd.DataFrame(rows)


def _layout_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layout_id, part in df.groupby("layout_id"):
        rows.append({"layout_id": layout_id, **_metrics(part)})
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["steel_true_kg"].to_numpy(dtype=float)
    pred = df["steel_pred_kg"].to_numpy(dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err**2)))
    r2 = float("nan")
    denom = float(np.sum((y - y.mean()) ** 2))
    if len(y) > 1 and denom > 0.0:
        r2 = float(1.0 - np.sum(err**2) / denom)
    return {
        "count": float(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": rmse,
        "mape": float(np.mean(abs_err / np.maximum(y, 1.0))),
        "r2": r2,
        "bias": float(err.mean()),
    }


def _qbucket(s: pd.Series) -> pd.Series:
    labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
    return pd.qcut(s.rank(method="first"), q=5, labels=labels)


def _value_or_quantile_bucket(s: pd.Series) -> pd.Series:
    if s.nunique(dropna=False) <= 12:
        return s.astype(str)
    return _qbucket(s)


def _plot_param_error_bars(param_metrics: pd.DataFrame, output_path: Path) -> None:
    params = PARAM_FEATURES
    ncols = 3
    nrows = int(np.ceil(len(params) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4.2 * nrows), constrained_layout=True)
    axes = axes.ravel()
    for ax, param in zip(axes, params):
        part = param_metrics[param_metrics["param"] == param].copy()
        part = part.sort_values("bucket")
        ax.bar(part["bucket"], part["mae"] / 1000.0, color="#4C78A8")
        ax.set_title(param)
        ax.set_ylabel("MAE (t)")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(params) :]:
        ax.axis("off")
    fig.suptitle("Steel prediction error by parameter bucket", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_true_steel_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Plot histogram of the true steel usage (`steel_true_kg`) and save to `output_path`."""
    vals = df["steel_true_kg"].dropna().to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(vals, bins=30, color="#4C78A8", edgecolor="black")
    ax.set_xlabel("True steel usage (kg)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of True Steel Usage")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _normalize_target(y: torch.Tensor, y_mean: float, y_std: float) -> torch.Tensor:
    return (torch.log1p(y) - y_mean) / y_std


def _plot_normalized_true_histogram(df: pd.DataFrame, output_path: Path) -> None:
    """Compute normalization using log1p, plot histogram of normalized targets, and save."""
    vals = df["steel_true_kg"].dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    y = torch.tensor(vals, dtype=torch.float32)
    y_log = torch.log1p(y)
    y_mean = float(y_log.mean().item())
    y_std = float(y_log.std(unbiased=False).item())
    norm = _normalize_target(y, y_mean, y_std).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(norm, bins=30, color="#4C78A8", edgecolor="black")
    ax.set_xlabel("Normalized target")
    ax.set_ylabel("Count")
    ax.set_title(f"Normalized True Steel Usage (mean={y_mean:.3f}, std={y_std:.3f})")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _make_report(
    summary: dict,
    y_bucket: pd.DataFrame,
    graph_tables: dict[str, pd.DataFrame],
    param_metrics: pd.DataFrame,
    layout_metrics: pd.DataFrame,
) -> str:
    lines = []
    lines.append("# Steel GNN Error Analysis")
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append(_one_row_table(pd.DataFrame([summary["overall"]])))
    lines.append("")
    lines.append("## By True Steel Usage Quantile")
    lines.append("")
    lines.append(_metric_table(y_bucket))
    lines.append("")
    lines.append("## By Graph Scale Quantile")
    for name, table in graph_tables.items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(_metric_table(table))
    lines.append("")
    lines.append("## Worst Parameter Buckets")
    lines.append("")
    worst_param = param_metrics.sort_values("mae", ascending=False).head(15)
    lines.append(
        _metric_table(worst_param[["param", "bucket", "count", "mae", "rmse", "mape", "r2", "bias"]])
    )
    lines.append("")
    lines.append("## Worst Layouts")
    lines.append("")
    lines.append(
        _metric_table(layout_metrics.head(15)[["layout_id", "count", "mae", "rmse", "mape", "r2", "bias"]])
    )
    lines.append("")
    lines.append("## Outputs")
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _metric_table(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in ["count"]:
        if col in out:
            out[col] = out[col].map(lambda v: f"{v:.0f}")
    for col in ["mae", "rmse", "bias", "range_min", "range_max"]:
        if col in out:
            out[col] = out[col].map(lambda v: f"{v:.1f}")
    for col in ["mape"]:
        if col in out:
            out[col] = out[col].map(lambda v: f"{100 * v:.2f}%")
    for col in ["r2"]:
        if col in out:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return _to_markdown(out)


def _one_row_table(df: pd.DataFrame) -> str:
    return _metric_table(df)


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[col]) for col in cols]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
