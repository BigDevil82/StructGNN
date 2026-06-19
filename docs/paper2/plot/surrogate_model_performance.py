from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOT_UTILS = ROOT / "docs" / "paper2" / "plot"
if str(PLOT_UTILS) not in sys.path:
    sys.path.insert(0, str(PLOT_UTILS))

from paper_plot_style import METHOD_COLORS, PANEL_CAPTION_SIZE, PLOT_DIR, save_figure, set_paper_style

GROUP_COLS = ["layout_id", "N", "hs", "h_story", "intensity", "site_class", "seismic_group"]
TARGET_RECALLS = [0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
MODEL_COLORS = {
    "Design-parameter MLP": "#8fb6d6",
    "Design-parameter + layout-statistics MLP": "#efbd75",
    "Room-graph LayoutParamGNN": "#e8a69d",
}


def main() -> None:
    args = build_parser().parse_args()
    set_paper_style()

    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    fig_dir = out_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_path)
    feasibility = load_feasibility_models(args.feasibility_models, dataset, args.screening_target_recall)
    steel = load_steel_models(
        args.steel_models,
        dataset,
        eval_pairs=args.eval_pairs,
        large_gap_kg=args.large_gap_kg,
        top_frac=args.top_frac,
        seed=args.seed,
    )

    feasibility["summary"].to_csv(table_dir / "feasibility_surrogate_metrics.csv", index=False)
    feasibility["screening_curve"].to_csv(table_dir / "feasibility_screening_curve.csv", index=False)
    feasibility["layout"].to_csv(table_dir / "feasibility_error_by_layout.csv", index=False)
    steel["summary"].to_csv(table_dir / "steel_surrogate_metrics.csv", index=False)
    steel["quantile"].to_csv(table_dir / "steel_error_by_true_quantile.csv", index=False)
    steel["layout"].to_csv(table_dir / "steel_error_by_layout.csv", index=False)

    plot_surrogate_summary(feasibility, steel, fig_dir, args.scatter_sample, args.seed)
    write_report(
        out_dir / "report.md",
        feasibility["summary"],
        feasibility["layout"],
        steel["summary"],
        steel["quantile"],
    )
    print(f"[paper2][surrogate] outputs written to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate paper Section 3.2 surrogate model performance tables and plots."
    )
    p.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    p.add_argument(
        "--feasibility-models",
        nargs="+",
        default=[
            r"Design-parameter MLP=data\parametric\ckpt\feas_ablation_param_only",
            r"Design-parameter + layout-statistics MLP=data\parametric\ckpt\feas_ablation_param_graph_feat",
            r"Room-graph LayoutParamGNN=data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1",
        ],
        help="Model specs as label=artifact_dir. Each artifact needs metrics.json and/or predictions_test.parquet.",
    )
    p.add_argument(
        "--steel-models",
        nargs="+",
        default=[
            r"Design-parameter MLP=data\parametric\ckpt\steel_ablation_param_only",
            r"Design-parameter + layout-statistics MLP=data\parametric\ckpt\steel_ablation_param_graph_feat",
            r"Room-graph LayoutParamGNN=data\parametric\ckpt\steel_gnn_room_lr5e4_b512",
        ],
        help="Model specs as label=artifact_dir. Each artifact needs predictions_test.parquet.",
    )
    p.add_argument("--out-dir", default=str(PLOT_DIR))
    p.add_argument("--screening-target-recall", type=float, default=0.995)
    p.add_argument("--eval-pairs", type=int, default=200000)
    p.add_argument("--large-gap-kg", type=float, default=10000.0)
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--scatter-sample", type=int, default=12000)
    p.add_argument("--seed", type=int, default=20260612)
    return p


def load_dataset(path: str) -> pd.DataFrame:
    cols = ["layout_id", "sample_id", "split", "final_pass", "material_steel_kg", *GROUP_COLS[1:]]
    df = pd.read_parquet(path, columns=cols)
    return df[df["split"] == "test"].copy()


def load_feasibility_models(
    specs: list[str], dataset: pd.DataFrame, target_recall: float
) -> dict[str, pd.DataFrame]:
    rows = []
    curves = []
    layout_rows = []
    best_df = None
    best_label = None
    best_reject = -1.0

    for label, model_dir in parse_model_specs(specs):
        pred = load_feasibility_predictions(model_dir, dataset)
        metrics = read_json(model_dir / "metrics.json")
        y = pred["y_true"].to_numpy(dtype=int)
        prob = pred["prob"].to_numpy(dtype=float)
        y_hat = prob >= float(metrics.get("threshold", 0.5))

        screen_threshold, screen_metrics = screening_at_target(pred, target_recall)
        artifact_screen = metrics.get("screening", {}).get("test", {})
        if artifact_screen:
            screen_threshold = float(artifact_screen.get("threshold", screen_threshold))
            screen_metrics = screening_metrics_at_threshold(pred, screen_threshold)

        cls = metrics.get("test", {})
        row = {
            "model": label,
            "samples": len(pred),
            "roc_auc": cls.get("roc_auc", safe_metric(roc_auc_score, y, prob)),
            "pr_auc": cls.get("pr_auc", safe_metric(average_precision_score, y, prob)),
            "precision": cls.get("precision", safe_metric(precision_score, y, y_hat, zero_division=0)),
            "recall": cls.get("recall", safe_metric(recall_score, y, y_hat, zero_division=0)),
            "f1": cls.get("f1", safe_metric(f1_score, y, y_hat, zero_division=0)),
            "screening_threshold": screen_threshold,
            **screen_metrics,
        }
        rows.append(row)

        curve = screening_curve(pred, TARGET_RECALLS)
        curve.insert(0, "model", label)
        curves.append(curve)
        layout_rows.append(feasibility_layout_metrics(pred, label))

        if row["reject_rate"] > best_reject:
            best_df = pred
            best_label = label
            best_reject = float(row["reject_rate"])

    summary = pd.DataFrame(rows)
    return {
        "summary": summary,
        "screening_curve": pd.concat(curves, ignore_index=True),
        "layout": pd.concat(layout_rows, ignore_index=True),
        "best_predictions": best_df,
        "best_label": best_label,
    }


def load_steel_models(
    specs: list[str],
    dataset: pd.DataFrame,
    *,
    eval_pairs: int,
    large_gap_kg: float,
    top_frac: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rows = []
    predictions = {}

    for label, model_dir in parse_model_specs(specs):
        df = load_steel_predictions(model_dir, dataset)
        y = df["y_true"].to_numpy(dtype=float)
        pred = df["y_pred"].to_numpy(dtype=float)
        rank = ranking_metrics(df, eval_pairs, large_gap_kg, top_frac, seed)
        rows.append(
            {
                "model": label,
                "samples": len(df),
                "mae": float(np.mean(np.abs(pred - y))),
                "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                "r2": safe_metric(r2_score, y, pred),
                "mape": float(np.mean(np.abs(pred - y) / np.maximum(y, 1.0))),
                "bias": float(np.mean(pred - y)),
                **rank,
            }
        )
        predictions[label] = df

    summary = pd.DataFrame(rows)
    best_label = str(summary.sort_values("mae").iloc[0]["model"])
    best_df = predictions[best_label]
    return {
        "summary": summary,
        "predictions": predictions,
        "best_label": best_label,
        "best_predictions": best_df,
        "quantile": steel_quantile_metrics(best_df, best_label),
        "layout": steel_layout_metrics(best_df, best_label),
    }


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    out = []
    for spec in specs:
        if "=" not in spec:
            path = Path(spec)
            out.append((path.name, path))
            continue
        label, path = spec.split("=", 1)
        out.append((label.strip(), Path(path.strip())))
    return out


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_feasibility_predictions(model_dir: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_parquet(model_dir / "predictions_test.parquet")
    prob_col = first_existing(pred, ["pred_final_pass_prob", "prob", "y_prob", "p_feasible"])
    true_col = first_existing(pred, ["final_pass_true", "final_pass", "y_true"])
    cols = ["layout_id", "sample_id", prob_col]
    if true_col:
        cols.append(true_col)
    out = pred[cols].copy()
    out = out.rename(columns={prob_col: "prob"})
    if true_col:
        out = out.rename(columns={true_col: "y_true"})
    else:
        out = out.merge(
            dataset[["layout_id", "sample_id", "final_pass"]], on=["layout_id", "sample_id"], how="left"
        )
        out = out.rename(columns={"final_pass": "y_true"})
    out["y_true"] = out["y_true"].astype(int)
    return out


def load_steel_predictions(model_dir: Path, dataset: pd.DataFrame) -> pd.DataFrame:
    pred = pd.read_parquet(model_dir / "predictions_test.parquet")
    true_col = first_existing(pred, ["steel_true_kg", "material_steel_kg", "y_true", "true", "target"])
    pred_col = first_existing(
        pred, ["steel_pred_kg", "pred_material_steel_kg", "y_pred", "pred", "prediction"]
    )
    if not pred_col:
        raise ValueError(f"No steel prediction column found in {model_dir}")

    cols = ["layout_id", "sample_id", pred_col]
    if true_col:
        cols.append(true_col)
    out = pred[cols].copy().rename(columns={pred_col: "y_pred"})
    if true_col:
        out = out.rename(columns={true_col: "y_true"})
    else:
        out = out.merge(
            dataset[["layout_id", "sample_id", "material_steel_kg"]],
            on=["layout_id", "sample_id"],
            how="left",
        )
        out = out.rename(columns={"material_steel_kg": "y_true"})
    meta = dataset[["layout_id", "sample_id", *GROUP_COLS[1:]]].copy()
    out = out.merge(meta, on=["layout_id", "sample_id"], how="left", validate="one_to_one")
    return out.dropna(subset=["y_true", "y_pred"]).copy()


def first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    for col in cols:
        if col in df.columns:
            return col
    return None


def screening_at_target(df: pd.DataFrame, target_recall: float) -> tuple[float, dict[str, float]]:
    feasible_prob = np.sort(df.loc[df["y_true"].astype(bool), "prob"].to_numpy(dtype=float))
    if len(feasible_prob) == 0:
        return 0.0, screening_metrics_at_threshold(df, 0.0)
    allowed_false = int(np.floor((1.0 - target_recall) * len(feasible_prob)))
    allowed_false = max(0, min(allowed_false, len(feasible_prob) - 1))
    threshold = float(np.nextafter(feasible_prob[allowed_false], -np.inf))
    return threshold, screening_metrics_at_threshold(df, threshold)


def screening_curve(df: pd.DataFrame, recalls: list[float]) -> pd.DataFrame:
    rows = []
    for target in recalls:
        threshold, metrics = screening_at_target(df, target)
        rows.append({"target_feasible_recall": target, "threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def screening_metrics_at_threshold(df: pd.DataFrame, threshold: float) -> dict[str, float]:
    y = df["y_true"].astype(bool).to_numpy()
    reject = df["prob"].to_numpy(dtype=float) < threshold
    keep = ~reject
    feasible = y
    infeasible = ~y
    false_reject = reject & feasible
    reject_infeasible = reject & infeasible
    kept_feasible = keep & feasible
    return {
        "reject_rate": float(reject.mean()),
        "keep_rate": float(keep.mean()),
        "feasible_recall": float(kept_feasible.sum() / max(feasible.sum(), 1)),
        "false_reject_rate": float(false_reject.sum() / max(feasible.sum(), 1)),
        "false_reject_count": int(false_reject.sum()),
        "reject_infeasible_precision": float(reject_infeasible.sum() / max(reject.sum(), 1)),
    }


def feasibility_layout_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for layout_id, part in df.groupby("layout_id"):
        y = part["y_true"].to_numpy(dtype=int)
        prob = part["prob"].to_numpy(dtype=float)
        rows.append(
            {
                "model": label,
                "layout_id": layout_id,
                "samples": len(part),
                "feasible_rate": float(y.mean()),
                "predicted_feasible_rate": float(prob.mean()),
                "calibration_gap": float(abs(prob.mean() - y.mean())),
                "brier": safe_metric(brier_score_loss, y, prob),
                "roc_auc": safe_binary_metric(roc_auc_score, y, prob),
                "pr_auc": safe_binary_metric(average_precision_score, y, prob),
            }
        )
    return pd.DataFrame(rows).sort_values("brier", ascending=False).reset_index(drop=True)


def ranking_metrics(
    df: pd.DataFrame, eval_pairs: int, large_gap_kg: float, top_frac: float, seed: int
) -> dict[str, float]:
    groups = {
        k: np.array(v, dtype=np.int64)
        for k, v in df.groupby(GROUP_COLS, observed=True).indices.items()
        if len(v) >= 2
    }
    if not groups:
        return {
            "pair_acc": float("nan"),
            "large_gap_acc": float("nan"),
            "spearman_group_mean": float("nan"),
            "top_recall": float("nan"),
            "regret_mean_kg": float("nan"),
        }

    y = df["y_true"].to_numpy(dtype=float)
    score = df["y_pred"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    i = np.empty(eval_pairs, dtype=np.int64)
    j = np.empty(eval_pairs, dtype=np.int64)
    for n in range(eval_pairs):
        group = groups[keys[int(rng.integers(0, len(keys)))]]
        a, b = rng.choice(group, size=2, replace=False)
        i[n] = a
        j[n] = b

    gap = np.abs(y[i] - y[j])
    ok = (score[i] < score[j]) == (y[i] < y[j])
    large = gap >= large_gap_kg
    spearman = []
    top_recall = []
    regrets = []
    for group in groups.values():
        if len(group) < 3:
            continue
        y_rank = pd.Series(y[group]).rank(method="average").to_numpy()
        s_rank = pd.Series(score[group]).rank(method="average").to_numpy()
        spearman.append(float(np.corrcoef(y_rank, s_rank)[0, 1]))

        k = max(1, int(np.ceil(len(group) * top_frac)))
        true_order = group[np.argsort(y[group])]
        pred_order = group[np.argsort(score[group])]
        top_recall.append(len(set(true_order[:k].tolist()) & set(pred_order[:k].tolist())) / k)
        regrets.append(float(y[pred_order[:k]].min() - y[true_order[0]]))

    return {
        "pair_acc": float(ok.mean()),
        "large_gap_acc": float(ok[large].mean()) if large.any() else float("nan"),
        "large_gap_rate": float(large.mean()),
        "spearman_group_mean": float(np.nanmean(spearman)),
        "top_recall": float(np.mean(top_recall)),
        "regret_mean_kg": float(np.mean(regrets)),
        "regret_p90_kg": float(np.quantile(regrets, 0.9)),
    }


def steel_quantile_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    work = df.copy()
    work["bucket"] = pd.qcut(
        work["y_true"].rank(method="first"),
        q=5,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    rows = []
    for bucket, part in work.groupby("bucket", observed=True):
        rows.append({"model": label, "bucket": str(bucket), **steel_metrics(part)})
    return pd.DataFrame(rows)


def steel_layout_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for layout_id, part in df.groupby("layout_id"):
        rows.append({"model": label, "layout_id": layout_id, **steel_metrics(part)})
    return pd.DataFrame(rows).sort_values("mae", ascending=False).reset_index(drop=True)


def steel_metrics(df: pd.DataFrame) -> dict[str, float]:
    y = df["y_true"].to_numpy(dtype=float)
    pred = df["y_pred"].to_numpy(dtype=float)
    err = pred - y
    return {
        "samples": len(df),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / np.maximum(y, 1.0))),
        "r2": safe_metric(r2_score, y, pred),
        "bias": float(np.mean(err)),
    }


def plot_surrogate_summary(
    feasibility: dict[str, pd.DataFrame],
    steel: dict[str, pd.DataFrame],
    out_dir: Path,
    sample: int,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    plot_feasibility_pr_curve(
        axes[0, 0], feasibility["best_predictions"], feasibility["summary"], feasibility["best_label"]
    )
    plot_feasibility_layout_calibration(axes[0, 1], feasibility["layout"], feasibility["best_label"])
    plot_steel_pred_true(axes[1, 0], steel["best_predictions"], steel["best_label"], sample, seed)
    plot_steel_quantile_error(axes[1, 1], steel["quantile"])
    captions = [
        "(a) Feasibility prediction: overall test performance",
        "(b) Feasibility prediction: layout-level probability bias",
        "(c) Steel usage prediction: overall test performance",
        "(d) Steel usage prediction: error by true-usage quantile",
    ]
    for ax, caption in zip(axes.ravel(), captions):
        add_panel_caption(ax, caption)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.98, wspace=0.22, hspace=0.35)
    save_figure(fig, "global_surrogate_overview.png", out_dir)


def add_panel_caption(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.5,
        -0.22,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_CAPTION_SIZE,
    )


def plot_feasibility_pr_curve(ax: plt.Axes, df: pd.DataFrame, summary: pd.DataFrame, label: str) -> None:
    y = df["y_true"].to_numpy(dtype=int)
    prob = df["prob"].to_numpy(dtype=float)
    precision, recall, _ = precision_recall_curve(y, prob)
    metrics = summary[summary["model"] == label].iloc[0]
    base_rate = float(y.mean())
    ax.plot(
        recall,
        precision,
        color=MODEL_COLORS.get(label, METHOD_COLORS["gnn_screen_cost"]),
        linewidth=2.0,
        label=label,
    )
    ax.axhline(base_rate, color="#777777", linewidth=1.0, linestyle="--", label="Base rate")
    ax.text(
        0.04,
        0.98,
        f"PR-AUC={metrics['pr_auc']:.3f}\nROC-AUC={metrics['roc_auc']:.3f}\nF1={metrics['f1']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=False)


def plot_feasibility_layout_calibration(ax: plt.Axes, layout: pd.DataFrame, label: str) -> None:
    part = layout[layout["model"] == label].copy()
    x = part["feasible_rate"].to_numpy(dtype=float)
    y = part["predicted_feasible_rate"].to_numpy(dtype=float)
    brier = part["brier"].to_numpy(dtype=float)
    sc = ax.scatter(
        x,
        y,
        c=brier,
        cmap="YlOrRd",
        s=34,
        alpha=0.85,
        edgecolor="#333333",
        linewidth=0.4,
    )
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=1.0, linestyle="--")
    ax.set_xlabel("True feasible rate by layout")
    ax.set_ylabel("Mean predicted probability")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Brier")


def plot_steel_pred_true(ax: plt.Axes, df: pd.DataFrame, label: str, sample: int, seed: int) -> None:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_pred - y_true)) / 1000.0)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)) / 1000.0)
    r2 = safe_metric(r2_score, y_true, y_pred)
    mape = float(np.mean(np.abs(y_pred - y_true) / np.maximum(y_true, 1.0)))
    part = df
    if len(part) > sample:
        part = part.sample(sample, random_state=seed)
    x = part["y_true"].to_numpy(dtype=float) / 1000.0
    y = part["y_pred"].to_numpy(dtype=float) / 1000.0
    ax.scatter(x, y, s=5, alpha=0.2, color=MODEL_COLORS.get(label, "#4C78A8"), edgecolors="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color="#555555", linewidth=1.0, linestyle="--")
    ax.text(
        0.04,
        0.96,
        f"R2={r2:.3f}\nMAE={mae:.1f} t\nRMSE={rmse:.1f} t\nMAPE={100 * mape:.1f}%",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    ax.set_xlabel("True steel usage (t)")
    ax.set_ylabel("Predicted steel usage (t)")
    ax.grid(True)


def plot_steel_quantile_error(ax: plt.Axes, quantile: pd.DataFrame) -> None:
    labels = quantile["bucket"].tolist()
    x = np.arange(len(labels))
    ax.bar(x, quantile["mae"] / 1000.0, color="#81b7ed", edgecolor="#ffffff", linewidth=0.8, label="MAE")
    ax.plot(x, quantile["bias"] / 1000.0, color="#d07c68", marker="o", linewidth=1.6, label="Bias")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("True steel usage quantile")
    ax.set_ylabel("Error (t)")
    ax.grid(axis="y")
    ax.legend(frameon=False)


def write_report(
    path: Path,
    feasibility: pd.DataFrame,
    feasibility_layout: pd.DataFrame,
    steel: pd.DataFrame,
    steel_quantile: pd.DataFrame,
) -> None:
    lines = [
        "# Surrogate Model Performance",
        "",
        "This report summarizes the experiments used for Section 3.2 of the paper outline.",
        "",
        "## Feasibility surrogate",
        "",
        to_markdown(format_feasibility(feasibility)),
        "",
        "### Worst layout-level feasibility probability errors",
        "",
        to_markdown(format_feasibility_layout(feasibility_layout.head(8))),
        "",
        "## Steel surrogate and representation comparison",
        "",
        to_markdown(format_steel(steel)),
        "",
        "### Steel error by true-usage quantile",
        "",
        to_markdown(format_steel_quantile(steel_quantile)),
        "",
        "Generated outputs:",
        "",
        "- `tables/feasibility_surrogate_metrics.csv`",
        "- `tables/feasibility_screening_curve.csv`",
        "- `tables/feasibility_error_by_layout.csv`",
        "- `tables/steel_surrogate_metrics.csv`",
        "- `tables/steel_error_by_true_quantile.csv`",
        "- `tables/steel_error_by_layout.csv`",
        "- `global_surrogate_overview.png`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def format_feasibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "reject_rate",
        "feasible_recall",
        "false_reject_rate",
        "reject_infeasible_precision",
    ]:
        if col in out:
            out[col] = out[col].map(lambda v: f"{float(v):.4f}")
    return out


def format_steel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["mae", "rmse", "bias", "regret_mean_kg", "regret_p90_kg"]:
        if col in out:
            out[col] = out[col].map(lambda v: f"{float(v):.1f}")
    for col in ["r2", "mape", "pair_acc", "large_gap_acc", "spearman_group_mean", "top_recall"]:
        if col in out:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


def format_feasibility_layout(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "layout_id",
        "samples",
        "feasible_rate",
        "predicted_feasible_rate",
        "calibration_gap",
        "brier",
    ]
    out = df[cols].copy()
    for col in ["feasible_rate", "predicted_feasible_rate", "calibration_gap", "brier"]:
        out[col] = out[col].map(lambda v: f"{float(v):.4f}")
    return out


def format_steel_quantile(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "bucket", "samples", "mae", "rmse", "mape", "r2", "bias"]
    out = df[cols].copy()
    for col in ["mae", "rmse", "bias"]:
        out[col] = out[col].map(lambda v: f"{float(v):.1f}")
    for col in ["mape", "r2"]:
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    return out


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(rows)


def safe_metric(fn, *args, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def safe_binary_metric(fn, y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return safe_metric(fn, y, score)


if __name__ == "__main__":
    main()
