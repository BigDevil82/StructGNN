from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from optimization_plot_data import (
    DEFAULT_EXPERIMENTS,
    complete_case_rows,
    find_result_json,
    load_all_summaries,
    material_points,
    paired_with_full,
    read_payload,
    screening_flow,
)
from paper_plot_style import (
    ALGORITHM_ORDER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LABELS_SHORT,
    METHOD_ORDER,
    PLOT_DIR,
    SURROGATE_METHODS,
    draw_violin_points,
    method_colors,
    save_figure,
    set_paper_style,
)


def main() -> None:
    args = build_parser().parse_args()
    set_paper_style()
    experiments = {
        "GA": Path(args.ga_dir),
        "PSO": Path(args.pso_dir),
        "Random Search": Path(args.random_dir),
    }
    df = load_all_summaries(experiments)
    paired = paired_with_full(df)

    plot_combined_distribution_summary(df, paired, args.out_dir)
    plot_tolerance_success_curve(paired, args.out_dir)
    plot_success_heatmap(df, args.out_dir)
    plot_process_examples(df, args.out_dir, args.process_algorithm)
    plot_screening_funnel(df, args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate paper-oriented figures for optimization experiments.")
    p.add_argument("--ga-dir", default=str(DEFAULT_EXPERIMENTS["GA"]))
    p.add_argument("--pso-dir", default=str(DEFAULT_EXPERIMENTS["PSO"]))
    p.add_argument("--random-dir", default=str(DEFAULT_EXPERIMENTS["Random Search"]))
    p.add_argument("--out-dir", default=str(PLOT_DIR))
    p.add_argument("--process-algorithm", choices=ALGORITHM_ORDER, default="GA")
    return p


def plot_combined_distribution_summary(df: pd.DataFrame, paired: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 7.2), sharex=False)
    fea_ymax = float(df["fea_calls"].max()) * 1.08
    first_ymax = float(df["first_feasible_fea_calls"].max(skipna=True)) * 1.08
    valid_gap = paired[paired["both_feasible"] & paired["cost_gap_pct"].notna()]
    gap_q = valid_gap["cost_gap_pct"].abs().quantile(0.98) if not valid_gap.empty else 10
    gap_lim = max(5.0, min(80.0, float(gap_q) * 1.2))

    for col_i, algorithm in enumerate(ALGORITHM_ORDER):
        sub = df[df["algorithm"] == algorithm]
        ax = axes[0, col_i]
        data = [sub.loc[sub["method"] == method, "fea_calls"].dropna().to_numpy() for method in METHOD_ORDER]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
            method_colors(METHOD_ORDER),
            ylabel="FEA calls" if col_i == 0 else None,
            title=algorithm,
            point_size=4,
        )
        ax.set_ylim(0, fea_ymax)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

        ax = axes[1, col_i]
        psub = paired[paired["algorithm"] == algorithm]
        data = [
            psub.loc[(psub["method"] == method) & psub["both_feasible"], "cost_gap_pct"].dropna().to_numpy()
            for method in SURROGATE_METHODS
        ]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in SURROGATE_METHODS],
            method_colors(SURROGATE_METHODS),
            ylabel="Cost gap (%)" if col_i == 0 else None,
            point_size=5,
        )
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_ylim(-gap_lim, gap_lim)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.set_xlabel("")

        ax = axes[2, col_i]
        data = [
            sub.loc[sub["method"] == method, "first_feasible_fea_calls"].dropna().to_numpy()
            for method in METHOD_ORDER
        ]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
            method_colors(METHOD_ORDER),
            ylabel="First feasible\nFEA calls" if col_i == 0 else None,
            point_size=5,
        )
        ax.set_ylim(0, first_ymax)

    for row_i, label in enumerate(["FEA effort", "Cost quality", "Feasible discovery"]):
        axes[row_i, 0].text(
            -0.24,
            0.5,
            label,
            transform=axes[row_i, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10,
            fontweight="bold",
        )
    fig.suptitle("Optimization Efficiency, Cost Quality, and Feasible Discovery", y=1.015)
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save_figure(fig, "main_01_distribution_summary_3x3.png", out_dir)


def plot_tolerance_success_curve(paired: pd.DataFrame, out_dir: str | Path) -> None:
    thresholds = np.array([0, 1, 2, 3, 5, 8, 10, 15], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        sub = paired[(paired["algorithm"] == algorithm) & paired["full_feasible"]].copy()
        for method in SURROGATE_METHODS:
            m = sub[sub["method"] == method]
            rates = []
            labels = []
            for threshold in thresholds:
                ok = m["best_feasible"] & (m["fea_reduction"] > 0) & (m["cost_gap_pct"] <= threshold)
                rates.append(ok.mean() if len(ok) else np.nan)
                labels.append(int(ok.sum()) if len(ok) else 0)
            ax.plot(
                thresholds,
                100 * np.asarray(rates, dtype=float),
                marker="o" if method == "gnn_cost" else "^",
                markersize=4.5,
                linewidth=1.8,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
            )
            for x, y, count in zip(thresholds, 100 * np.asarray(rates, dtype=float), labels):
                if np.isfinite(y):
                    ax.text(x, y + 1.5, str(count), ha="center", va="bottom", fontsize=6, color="#444444")
        ax.set_title(algorithm)
        ax.set_xlabel("Allowed cost increase vs full (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(thresholds.min(), thresholds.max())
        ax.grid(True)
        ax.set_ylabel("Qualified ratio among full-feasible cases (%)")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle("Efficiency-Quality Qualification Under Cost Tolerance", y=1.02)
    fig.tight_layout()
    save_figure(fig, "main_02_tolerance_success_curve.png", out_dir)


def plot_success_heatmap(df: pd.DataFrame, out_dir: str | Path) -> None:
    df = df.copy()
    df["col"] = df["algorithm"] + "\n" + df["method"].map(METHOD_LABELS_SHORT)
    col_order = [f"{algo}\n{METHOD_LABELS_SHORT[m]}" for algo in ALGORITHM_ORDER for m in METHOD_ORDER]
    grouped = df.groupby(["layout_id", "col"])["best_feasible"].agg(["mean", "sum", "count"]).reset_index()
    matrix = grouped.pivot(index="layout_id", columns="col", values="mean")[col_order]
    layout_order = _sort_layouts_by_success_pattern(matrix, col_order)
    matrix = matrix.reindex(layout_order)
    counts = grouped.pivot(index="layout_id", columns="col", values="sum").reindex(layout_order)[col_order]
    totals = grouped.pivot(index="layout_id", columns="col", values="count").reindex(layout_order)[col_order]

    fig, ax = plt.subplots(figsize=(12.5, max(5.0, 0.28 * len(layout_order))))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layout_order)))
    ax.set_yticklabels(layout_order)
    ax.set_title("Feasible Solution Success Rate by Layout")
    for i, layout in enumerate(layout_order):
        for j, col in enumerate(col_order):
            value = matrix.loc[layout, col]
            if pd.isna(value):
                continue
            text_color = "#ffffff" if value >= 0.65 else "#222222"
            ax.text(
                j,
                i,
                f"{int(counts.loc[layout, col])}/{int(totals.loc[layout, col])}",
                ha="center",
                va="center",
                fontsize=6,
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Success rate")
    fig.tight_layout()
    save_figure(fig, "main_03_success_heatmap.png", out_dir)


def _sort_layouts_by_success_pattern(matrix: pd.DataFrame, col_order: list[str]) -> list[str]:
    sort_df = matrix[col_order].fillna(-1.0).copy()
    sort_df["_layout_id"] = sort_df.index.astype(str)
    sort_df = sort_df.sort_values(col_order + ["_layout_id"], ascending=[False] * len(col_order) + [True])
    return sort_df.index.tolist()


def plot_process_examples(df: pd.DataFrame, out_dir: str | Path, algorithm: str) -> None:
    selected = select_representative_cases(df, algorithm)
    fig, axes = plt.subplots(len(selected), 3, figsize=(12.0, 2.35 * len(selected)), sharey="row")
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_i, case in enumerate(selected):
        case_rows = df[
            (df["algorithm"] == algorithm)
            & (df["layout_id"] == case["layout_id"])
            & (df["seed"] == case["seed"])
        ]
        ymin, ymax = collect_case_ylim(case_rows)
        for col_i, method in enumerate(METHOD_ORDER):
            ax = axes[row_i, col_i]
            row = case_rows[case_rows["method"] == method]
            if row.empty:
                empty_panel(ax, "No result")
                continue
            path = find_result_json(row.iloc[0])
            if path is None:
                empty_panel(ax, "Missing JSON")
                continue
            points = material_points(read_payload(path).get("history", []))
            draw_process_panel(ax, points, method)
            ax.set_ylim(ymin, ymax)
            format_million_axis(ax)
            if row_i == 0:
                ax.set_title(METHOD_LABELS[method])
            if col_i == 0:
                ax.set_ylabel(f"{case['label']}\n{case['layout_id']} seed {int(case['seed'])}\nMaterial cost")
            else:
                ax.set_ylabel("")
            if row_i == len(selected) - 1:
                ax.set_xlabel("Evaluated candidate")
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Representative {algorithm} Search Trajectories", y=1.04)
    fig.tight_layout()
    save_figure(fig, "main_04_process_scatter_examples.png", out_dir)


def plot_screening_funnel(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    flow_colors = ["#e8a69d", "#efbd75", "#8fb6d6"]
    flow_labels = ["Feasibility rejected", "Cost skipped", "FEA evaluated"]
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        rows = df[(df["algorithm"] == algorithm) & (df["method"] == "gnn_screen_cost")]
        flow = aggregate_flow(rows)
        if flow.empty:
            empty_panel(ax, "No flow data")
            continue
        x = flow["step"].to_numpy()
        ax.stackplot(
            x,
            flow["screened"],
            flow["cost_skip"],
            flow["fea"],
            colors=flow_colors,
            labels=flow_labels,
            alpha=0.82,
        )
        ax.plot(x, flow["feasible"], color="#2f7d32", linewidth=1.6, label="FEA feasible")
        ax.set_title(algorithm)
        ax.set_xlabel("Generation / batch")
        ax.set_ylim(0, 1)
        ax.grid(axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Mean candidate fraction")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Candidate Flow in the Screening + Cost Surrogate Strategy", y=1.02)
    fig.tight_layout()
    save_figure(fig, "supp_03_screening_funnel.png", out_dir)


def select_representative_cases(df: pd.DataFrame, algorithm: str) -> list[dict[str, object]]:
    rows = complete_case_rows(df, algorithm)
    full = rows[rows["method"] == "full"].copy()
    feasible = full[full["best_feasible"] & full["first_feasible_fea_calls"].notna()].sort_values(
        "first_feasible_fea_calls"
    )
    selected: list[dict[str, object]] = []
    used_layouts: set[str] = set()
    if not feasible.empty:
        easy = feasible.iloc[0]
        selected.append(case_dict(easy, "Easy"))
        used_layouts.add(str(easy["layout_id"]))
        medium = _pick_distinct_case(feasible, len(feasible) // 2, used_layouts)
        if medium is not None:
            selected.append(case_dict(medium, "Medium"))
            used_layouts.add(str(medium["layout_id"]))
    hard = full[~full["best_feasible"]].sort_values("best_objective")
    if not hard.empty:
        row = _pick_distinct_case(hard, 0, used_layouts)
        selected.append(case_dict(row if row is not None else hard.iloc[0], "Hard"))
    elif len(feasible) >= 3:
        row = _pick_distinct_case(feasible, len(feasible) - 1, used_layouts)
        selected.append(case_dict(row if row is not None else feasible.iloc[-1], "Hard"))
    return selected[:3]


def _pick_distinct_case(df: pd.DataFrame, target_idx: int, used_layouts: set[str]) -> pd.Series | None:
    if df.empty:
        return None
    target_idx = min(max(int(target_idx), 0), len(df) - 1)
    order = sorted(range(len(df)), key=lambda i: (abs(i - target_idx), i))
    for i in order:
        row = df.iloc[i]
        if str(row["layout_id"]) not in used_layouts:
            return row
    return df.iloc[target_idx]


def case_dict(row: pd.Series, label: str) -> dict[str, object]:
    return {"layout_id": row["layout_id"], "seed": row["seed"], "label": label}


def collect_case_ylim(case_rows: pd.DataFrame) -> tuple[float, float]:
    values: list[pd.Series] = []
    for _, row in case_rows.iterrows():
        path = find_result_json(row)
        if path is None:
            continue
        points = material_points(read_payload(path).get("history", []))
        if not points.empty:
            values.append(points["material_cost"])
    if not values:
        return (0.0, 1.0)

    all_values = pd.concat(values, ignore_index=True).dropna()
    if all_values.empty:
        return (0.0, 1.0)

    low = float(all_values.quantile(0.02))
    high = float(all_values.quantile(0.98))
    if high <= low:
        high = float(all_values.max())
        low = float(all_values.min())
    pad = max((high - low) * 0.10, high * 0.02, 1.0)
    return (max(0.0, low - pad), high + pad)


def format_million_axis(ax: plt.Axes) -> None:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_powerlimits((6, 6))
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))


def draw_process_panel(ax: plt.Axes, points: pd.DataFrame, method: str) -> None:
    if points.empty:
        empty_panel(ax, "No evaluated candidates")
        return
    infeasible = points[~points["feasible"]]
    feasible = points[points["feasible"]]
    ax.scatter(
        infeasible["evaluation"],
        infeasible["material_cost"],
        s=10,
        color="#b9b9b9",
        edgecolor="none",
        alpha=0.45,
        label="Infeasible",
    )
    ax.scatter(
        feasible["evaluation"],
        feasible["material_cost"],
        s=12,
        color="#4e9f65",
        edgecolor="#1f5c35",
        linewidth=0.2,
        alpha=0.75,
        label="Feasible",
    )
    line = points.dropna(subset=["best_feasible_cost"])
    if not line.empty:
        ax.plot(
            line["evaluation"],
            line["best_feasible_cost"],
            color=METHOD_COLORS[method],
            linewidth=1.4,
            label="Best feasible",
        )
    ax.grid(True)


def aggregate_flow(rows: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, row in rows.iterrows():
        path = find_result_json(row)
        if path is None:
            continue
        flow = screening_flow(read_payload(path).get("history", []))
        if not flow.empty:
            frames.append(flow)
    if not frames:
        return pd.DataFrame()
    flow = pd.concat(frames, ignore_index=True)
    return flow.groupby("step")[["screened", "cost_skip", "fea", "feasible"]].mean().reset_index()


def empty_panel(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    main()
