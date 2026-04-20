from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve


@dataclass(frozen=True)
class PredictionAnalysisConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    prediction_path: str = r"data\parametric\surrogate_dataset\predictions_lightgbm.parquet"
    output_dir: str = r"outputs\surrogate\analysis_figures"
    scatter_sample_size: int = 10000
    random_seed: int = 42


def generate_prediction_analysis_figures(cfg: PredictionAnalysisConfig) -> dict[str, object]:
    data = pd.read_parquet(cfg.dataset_path)
    pred = pd.read_parquet(cfg.prediction_path)

    merged = data.merge(pred, on=["layout_id", "sample_id"], how="inner", validate="one_to_one")
    test_df = merged[merged["split"] == "test"].copy().reset_index(drop=True)

    analysis_df = _build_analysis_table(test_df)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_table_path = out_dir / "test_predictions_with_residuals.parquet"
    analysis_df.to_parquet(analysis_table_path, index=False)

    layout_df = _build_layout_metrics(analysis_df)
    layout_table_path = out_dir / "layout_level_metrics.parquet"
    layout_df.to_parquet(layout_table_path, index=False)

    _plot_pr_curve(analysis_df, out_dir / "fig01_pr_curve.png")
    _plot_probability_hist(analysis_df, out_dir / "fig02_prob_distribution.png")

    sampled = _sample_for_scatter(analysis_df, cfg.scatter_sample_size, cfg.random_seed)
    _plot_true_pred(
        sampled, "max_drift_ratio_true", "max_drift_ratio_pred", out_dir / "fig03_drift_true_vs_pred.png"
    )
    _plot_true_pred(
        sampled,
        "material_steel_kg_true",
        "material_steel_kg_pred",
        out_dir / "fig04_steel_true_vs_pred.png",
    )
    _plot_true_pred(
        sampled, "torsion_ratio_true", "torsion_ratio_pred", out_dir / "fig05_torsion_true_vs_pred.png"
    )

    _plot_residual(
        sampled, "max_drift_ratio_true", "max_drift_ratio_residual", out_dir / "fig06_drift_residual.png"
    )
    _plot_residual(
        sampled, "torsion_ratio_true", "torsion_ratio_residual", out_dir / "fig07_torsion_residual.png"
    )

    _plot_layout_scatter(
        layout_df,
        "layout_positive_rate",
        "layout_f1",
        out_dir / "fig08_layout_positive_rate_vs_f1.png",
        x_label="layout_positive_rate",
        y_label="layout_f1",
    )

    fig09_path = out_dir / "fig09_ecc_center_vs_layout_mae_torsion.png"
    has_ecc = "ecc_center" in layout_df.columns and layout_df["ecc_center"].notna().any()
    if has_ecc:
        _plot_layout_scatter(
            layout_df,
            "ecc_center",
            "layout_mae_torsion",
            fig09_path,
            x_label="ecc_center",
            y_label="layout_mae_torsion",
        )

    _plot_torsion_top_bottom(layout_df, out_dir / "fig10_torsion_top_bottom_layouts.png")

    generated = [
        "fig01_pr_curve.png",
        "fig02_prob_distribution.png",
        "fig03_drift_true_vs_pred.png",
        "fig04_steel_true_vs_pred.png",
        "fig05_torsion_true_vs_pred.png",
        "fig06_drift_residual.png",
        "fig07_torsion_residual.png",
        "fig08_layout_positive_rate_vs_f1.png",
        "fig10_torsion_top_bottom_layouts.png",
    ]
    if has_ecc:
        generated.append("fig09_ecc_center_vs_layout_mae_torsion.png")

    return {
        "output_dir": str(out_dir),
        "test_rows": int(len(analysis_df)),
        "layout_rows": int(len(layout_df)),
        "analysis_table": str(analysis_table_path),
        "layout_table": str(layout_table_path),
        "generated_figures": generated,
        "skipped_fig09_due_to_missing_ecc_center": not has_ecc,
    }


def _build_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["final_pass_true"] = out["final_pass"].astype(int)
    out["final_pass_prob"] = out["pred_final_pass_prob"].astype(float)
    out["final_pass_pred"] = out["pred_final_pass"].astype(int)

    out["max_drift_ratio_true"] = out["max_drift_ratio"].astype(float)
    out["max_drift_ratio_pred"] = out["pred_max_drift_ratio"].astype(float)

    out["torsion_ratio_true"] = out["torsion_ratio"].astype(float)
    out["torsion_ratio_pred"] = out["pred_torsion_ratio"].astype(float)

    out["material_steel_kg_true"] = out["material_steel_kg"].astype(float)
    out["material_steel_kg_pred"] = out["pred_material_steel_kg"].astype(float)

    out["max_drift_ratio_residual"] = out["max_drift_ratio_pred"] - out["max_drift_ratio_true"]
    out["torsion_ratio_residual"] = out["torsion_ratio_pred"] - out["torsion_ratio_true"]
    out["material_steel_kg_residual"] = out["material_steel_kg_pred"] - out["material_steel_kg_true"]

    keep_cols = [
        "layout_id",
        "sample_id",
        "final_pass_true",
        "final_pass_prob",
        "final_pass_pred",
        "max_drift_ratio_true",
        "max_drift_ratio_pred",
        "torsion_ratio_true",
        "torsion_ratio_pred",
        "material_steel_kg_true",
        "material_steel_kg_pred",
        "max_drift_ratio_residual",
        "torsion_ratio_residual",
        "material_steel_kg_residual",
    ]
    if "ecc_center" in out.columns:
        keep_cols.append("ecc_center")

    return out[keep_cols].copy()


def _build_layout_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for layout_id, grp in df.groupby("layout_id"):
        f1 = f1_score(grp["final_pass_true"], grp["final_pass_pred"], zero_division=0)
        row = {
            "layout_id": layout_id,
            "layout_positive_rate": float(grp["final_pass_true"].mean()),
            "layout_f1": float(f1),
            "layout_mae_torsion": float(
                np.mean(np.abs(grp["torsion_ratio_pred"] - grp["torsion_ratio_true"]))
            ),
        }
        if "ecc_center" in grp.columns:
            row["ecc_center"] = float(grp["ecc_center"].iloc[0])
        records.append(row)
    return pd.DataFrame(records).sort_values("layout_id").reset_index(drop=True)


def _sample_for_scatter(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if len(df) <= size:
        return df
    return df.sample(n=size, random_state=seed)


def _plot_pr_curve(df: pd.DataFrame, out_path: Path) -> None:
    y_true = df["final_pass_true"].to_numpy()
    y_prob = df["final_pass_prob"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Final Pass PR Curve (PR-AUC={pr_auc:.4f})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_probability_hist(df: pd.DataFrame, out_path: Path) -> None:
    pos = df[df["final_pass_true"] == 1]["final_pass_prob"]
    neg = df[df["final_pass_true"] == 0]["final_pass_prob"]

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.5, label="true=0", density=True)
    plt.hist(pos, bins=40, alpha=0.5, label="true=1", density=True)
    plt.xlabel("final_pass_prob")
    plt.ylabel("Density")
    plt.title("Final Pass Probability Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_true_pred(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path) -> None:
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))

    plt.figure(figsize=(6.5, 6))
    plt.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1)
    plt.colorbar(label="count")
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_residual(df: pd.DataFrame, x_col: str, y_col: str, out_path: Path) -> None:
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.hexbin(x, y, gridsize=60, cmap="magma", mincnt=1)
    plt.colorbar(label="count")
    plt.axhline(0.0, color="r", linestyle="--", linewidth=1.2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_layout_scatter(
    layout_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    x_label: str,
    y_label: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(layout_df[x_col], layout_df[y_col], alpha=0.8)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_torsion_top_bottom(layout_df: pd.DataFrame, out_path: Path) -> None:
    sorted_df = layout_df.sort_values("layout_mae_torsion")
    top10 = sorted_df.tail(10)
    bottom10 = sorted_df.head(10)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=False)

    axes[0].bar(top10["layout_id"], top10["layout_mae_torsion"], color="#c44e52")
    axes[0].set_title("Top 10 Worst Layouts by Torsion MAE")
    axes[0].set_ylabel("layout_mae_torsion")
    axes[0].tick_params(axis="x", labelrotation=45)

    axes[1].bar(bottom10["layout_id"], bottom10["layout_mae_torsion"], color="#55a868")
    axes[1].set_title("Top 10 Best Layouts by Torsion MAE")
    axes[1].set_ylabel("layout_mae_torsion")
    axes[1].tick_params(axis="x", labelrotation=45)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
