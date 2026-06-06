from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from optimization_plot_data import (
    DEFAULT_EXPERIMENTS,
    complete_case_rows,
    find_result_json,
    load_all_summaries,
    material_points,
    paired_with_full,
    read_payload,
    screening_flow,
    sort_layouts_for_heatmap,
)
from paper_plot_style import (
    ALGORITHM_ORDER,
    FAMILY_MARKERS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_LABELS_SHORT,
    METHOD_ORDER,
    PLOT_DIR,
    SURROGATE_METHODS,
    add_panel_label,
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

    plot_fea_calls(df, args.out_dir)
    plot_efficiency_quality(paired, args.out_dir)
    plot_paired_delta(paired, args.out_dir)
    plot_tolerance_success_curve(paired, args.out_dir)
    plot_success_heatmap(df, args.out_dir)
    plot_process_examples(df, args.out_dir, args.process_algorithm)
    plot_cost_gap(paired, args.out_dir)
    plot_first_feasible(df, args.out_dir)
    plot_screening_funnel(df, args.out_dir)
    plot_difficulty_benefit(df, paired, args.out_dir)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate paper-oriented figures for optimization experiments.")
    p.add_argument("--ga-dir", default=str(DEFAULT_EXPERIMENTS["GA"]))
    p.add_argument("--pso-dir", default=str(DEFAULT_EXPERIMENTS["PSO"]))
    p.add_argument("--random-dir", default=str(DEFAULT_EXPERIMENTS["Random Search"]))
    p.add_argument("--out-dir", default=str(PLOT_DIR))
    p.add_argument("--process-algorithm", choices=ALGORITHM_ORDER, default="GA")
    return p


def plot_fea_calls(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    ymax = float(df["fea_calls"].max())
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        sub = df[df["algorithm"] == algorithm]
        data = [sub.loc[sub["method"] == method, "fea_calls"].dropna().to_numpy() for method in METHOD_ORDER]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
            method_colors(METHOD_ORDER),
            ylabel="FEA calls" if ax is axes[0] else None,
            title=algorithm,
            point_size=5,
        )
        ax.set_ylim(0, ymax * 1.08)
    fig.suptitle("FEA Calls by Optimization Algorithm and Surrogate Strategy", y=1.02)
    fig.tight_layout()
    save_figure(fig, "main_01_fea_calls_violin.png", out_dir)


def plot_efficiency_quality(paired: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    valid = paired[paired["both_feasible"] & paired["cost_ratio"].notna()]
    ymax = max(1.25, float(valid["cost_ratio"].quantile(0.98)) * 1.05) if not valid.empty else 1.25
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        sub = paired[paired["algorithm"] == algorithm]
        for method in SURROGATE_METHODS:
            m = sub[sub["method"] == method]
            ok = m[m["both_feasible"] & m["cost_ratio"].notna()]
            ax.scatter(
                ok["fea_ratio"],
                ok["cost_ratio"],
                s=26,
                color=METHOD_COLORS[method],
                edgecolor="#222222",
                linewidth=0.35,
                alpha=0.72,
                label=METHOD_LABELS[method],
            )
            failed = m[m["full_feasible"] & ~m["best_feasible"]]
            if not failed.empty:
                ax.scatter(
                    failed["fea_ratio"],
                    np.full(len(failed), ymax),
                    marker="x",
                    s=24,
                    color=METHOD_COLORS[method],
                    linewidth=0.9,
                    alpha=0.9,
                )
        ax.axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        ax.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_title(algorithm)
        ax.set_xlabel("FEA ratio to full")
        ax.set_xlim(0, 1.08)
        ax.set_ylim(0.75, ymax * 1.02)
        ax.grid(True)
        if ax is axes[0]:
            ax.set_ylabel("Cost ratio to full")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Efficiency-Quality Tradeoff Relative to Full FEA", y=1.02)
    fig.tight_layout()
    save_figure(fig, "main_02_efficiency_quality_pareto.png", out_dir)


def plot_paired_delta(paired: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.2), sharex="col")
    for row_i, algorithm in enumerate(ALGORITHM_ORDER):
        sub = paired[paired["algorithm"] == algorithm].copy()
        sub = sub[sub["full_feasible"]].copy()
        order = case_order(sub)
        for col_i, metric in enumerate(["fea_reduction", "cost_gap_pct"]):
            ax = axes[row_i, col_i]
            metric_df = sub[sub["case_id"].isin(order)]
            for y, case_id in enumerate(order):
                rows = metric_df[metric_df["case_id"] == case_id]
                values = rows.set_index("method")[metric].to_dict()
                xs = [values.get(method, np.nan) for method in SURROGATE_METHODS]
                if all(np.isfinite(xs)):
                    ax.plot(xs, [y, y], color="#d2d2d2", linewidth=0.8, zorder=1)
                for method in SURROGATE_METHODS:
                    row = rows[rows["method"] == method]
                    if row.empty:
                        continue
                    x = float(row.iloc[0][metric])
                    if not np.isfinite(x):
                        continue
                    marker = "o" if method == "gnn_cost" else "^"
                    ax.scatter(
                        x,
                        y,
                        s=30,
                        marker=marker,
                        color=METHOD_COLORS[method],
                        edgecolor="#222222",
                        linewidth=0.35,
                        alpha=0.85,
                        zorder=2,
                    )
            ax.set_title(f"{algorithm} - {'FEA reduction' if col_i == 0 else 'Cost gap'}")
            ax.set_yticks([])
            ax.set_ylim(-1, len(order))
            ax.grid(axis="x")
            if col_i == 0:
                ax.set_xlim(-0.05, 1.02)
                ax.set_xlabel("FEA reduction vs full")
            else:
                ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
                ax.set_xlim(delta_xlim(sub["cost_gap_pct"]))
                ax.set_xlabel("Cost gap vs full (%)")
            if col_i == 0:
                ax.set_ylabel(f"{algorithm}\npaired cases")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS["gnn_cost"],
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            markersize=6,
            label=METHOD_LABELS["gnn_cost"],
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=METHOD_COLORS["gnn_screen_cost"],
            markeredgecolor="#222222",
            markeredgewidth=0.35,
            markersize=6,
            label=METHOD_LABELS["gnn_screen_cost"],
        ),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Paired Case-Level Changes Relative to Full FEA", y=1.035)
    fig.tight_layout()
    save_figure(fig, "candidate_02a_paired_delta.png", out_dir)


def plot_tolerance_success_curve(paired: pd.DataFrame, out_dir: str | Path) -> None:
    thresholds = np.array([0, 1, 2, 3, 5, 8, 10, 15], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
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
        if ax is axes[0]:
            ax.set_ylabel("Qualified case ratio (%)")
    axes[-1].legend(loc="lower right", frameon=False)
    fig.suptitle("Efficiency-Quality Qualification Under Cost Tolerance", y=1.02)
    fig.tight_layout()
    save_figure(fig, "candidate_02b_tolerance_success_curve.png", out_dir)


def plot_success_heatmap(df: pd.DataFrame, out_dir: str | Path) -> None:
    df = df.copy()
    df["col"] = df["algorithm"] + "\n" + df["method"].map(METHOD_LABELS_SHORT)
    layout_order = sort_layouts_for_heatmap(df)
    col_order = [f"{algo}\n{METHOD_LABELS_SHORT[m]}" for algo in ALGORITHM_ORDER for m in METHOD_ORDER]
    grouped = df.groupby(["layout_id", "col"])["best_feasible"].agg(["mean", "sum", "count"]).reset_index()
    matrix = grouped.pivot(index="layout_id", columns="col", values="mean").reindex(layout_order)[col_order]
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


def plot_process_examples(df: pd.DataFrame, out_dir: str | Path, algorithm: str) -> None:
    selected = select_representative_cases(df, algorithm)
    fig, axes = plt.subplots(len(selected), 3, figsize=(12.0, 3.0 * len(selected)), sharey="row")
    if len(selected) == 1:
        axes = np.asarray([axes])

    for row_i, case in enumerate(selected):
        case_rows = df[
            (df["algorithm"] == algorithm)
            & (df["layout_id"] == case["layout_id"])
            & (df["seed"] == case["seed"])
        ]
        ymax = collect_case_ymax(case_rows)
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
            ax.set_ylim(0, ymax * 1.05 if ymax > 0 else 1)
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
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Representative {algorithm} Search Trajectories", y=1.04)
    fig.tight_layout()
    save_figure(fig, "main_04_process_scatter_examples.png", out_dir)


def plot_cost_gap(paired: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    valid = paired[paired["both_feasible"] & paired["cost_gap_pct"].notna()]
    q = valid["cost_gap_pct"].abs().quantile(0.98) if not valid.empty else 10
    lim = max(5.0, min(80.0, float(q) * 1.2))
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        sub = paired[paired["algorithm"] == algorithm]
        data = [
            sub.loc[(sub["method"] == method) & sub["both_feasible"], "cost_gap_pct"].dropna().to_numpy()
            for method in SURROGATE_METHODS
        ]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in SURROGATE_METHODS],
            method_colors(SURROGATE_METHODS),
            ylabel="Cost gap to full (%)" if ax is axes[0] else None,
            title=algorithm,
            point_size=9,
        )
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_ylim(-lim, lim)
    fig.suptitle("Final Cost Gap Relative to Full FEA", y=1.02)
    fig.tight_layout()
    save_figure(fig, "supp_01_cost_gap_violin.png", out_dir)


def plot_first_feasible(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=True)
    ymax = float(df["first_feasible_fea_calls"].max(skipna=True))
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        sub = df[df["algorithm"] == algorithm]
        data = [sub.loc[sub["method"] == method, "first_feasible_fea_calls"].dropna().to_numpy() for method in METHOD_ORDER]
        labels = []
        for method in METHOD_ORDER:
            m = sub[sub["method"] == method]
            labels.append(f"{METHOD_LABELS_SHORT[method]}\nfail={int((~m['best_feasible']).sum())}")
        draw_violin_points(
            ax,
            data,
            labels,
            method_colors(METHOD_ORDER),
            ylabel="FEA calls to first feasible" if ax is axes[0] else None,
            title=algorithm,
            point_size=8,
        )
        ax.set_ylim(0, ymax * 1.08)
    fig.suptitle("Search Effort Before the First Feasible Design", y=1.02)
    fig.tight_layout()
    save_figure(fig, "supp_02_first_feasible_violin.png", out_dir)


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


def plot_difficulty_benefit(df: pd.DataFrame, paired: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        full = (
            df[(df["algorithm"] == algorithm) & (df["method"] == "full")]
            .groupby(["layout_id", "family"])["best_feasible"]
            .mean()
            .reset_index(name="full_success_rate")
        )
        benefit = (
            paired[paired["algorithm"] == algorithm]
            .groupby(["layout_id", "family", "method"])["fea_reduction"]
            .median()
            .reset_index()
        )
        merged = benefit.merge(full, on=["layout_id", "family"], how="left")
        for method in SURROGATE_METHODS:
            for family, marker in FAMILY_MARKERS.items():
                sub = merged[(merged["method"] == method) & (merged["family"] == family)]
                if sub.empty:
                    continue
                ax.scatter(
                    sub["full_success_rate"],
                    100 * sub["fea_reduction"],
                    s=36,
                    marker=marker,
                    color=METHOD_COLORS[method],
                    edgecolor="#222222",
                    linewidth=0.35,
                    alpha=0.78,
                    label=f"{METHOD_LABELS_SHORT[method]} / {family}",
                )
        ax.set_title(algorithm)
        ax.set_xlabel("Full FEA success rate by layout")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 105)
        ax.grid(True)
        if ax is axes[0]:
            ax.set_ylabel("Median FEA reduction (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Layout Difficulty and Surrogate Evaluation Savings", y=1.11)
    fig.tight_layout()
    save_figure(fig, "supp_04_difficulty_vs_benefit.png", out_dir)


def select_representative_cases(df: pd.DataFrame, algorithm: str) -> list[dict[str, object]]:
    rows = complete_case_rows(df, algorithm)
    full = rows[rows["method"] == "full"].copy()
    feasible = full[full["best_feasible"] & full["first_feasible_fea_calls"].notna()].sort_values(
        "first_feasible_fea_calls"
    )
    selected: list[dict[str, object]] = []
    if not feasible.empty:
        selected.append(case_dict(feasible.iloc[0], "Easy"))
        selected.append(case_dict(feasible.iloc[len(feasible) // 2], "Medium"))
    hard = full[~full["best_feasible"]].sort_values("best_objective")
    if not hard.empty:
        selected.append(case_dict(hard.iloc[0], "Hard"))
    elif len(feasible) >= 3:
        selected.append(case_dict(feasible.iloc[-1], "Hard"))
    return selected[:3]


def case_dict(row: pd.Series, label: str) -> dict[str, object]:
    return {"layout_id": row["layout_id"], "seed": row["seed"], "label": label}


def collect_case_ymax(case_rows: pd.DataFrame) -> float:
    ymax = 0.0
    for _, row in case_rows.iterrows():
        path = find_result_json(row)
        if path is None:
            continue
        points = material_points(read_payload(path).get("history", []))
        if not points.empty:
            ymax = max(ymax, float(points["material_cost"].quantile(0.98)))
    return ymax


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


def case_order(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return []
    stats = (
        df.groupby("case_id")
        .agg(
            family=("family", "first"),
            layout_id=("layout_id", "first"),
            seed=("seed", "first"),
            fea_reduction=("fea_reduction", "median"),
            cost_gap_pct=("cost_gap_pct", "median"),
        )
        .reset_index()
    )
    stats["family_order"] = stats["family"].map({"L17": 0, "L27": 1, "L1L28": 2}).fillna(9)
    stats["layout_num"] = stats["layout_id"].str.extract(r"_(\d+)").astype(float)
    stats = stats.sort_values(["family_order", "layout_num", "seed"])
    return stats["case_id"].tolist()


def delta_xlim(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna()
    if clean.empty:
        return (-10.0, 10.0)
    q = float(clean.abs().quantile(0.98))
    lim = max(5.0, min(50.0, q * 1.25))
    return (-lim, lim)


if __name__ == "__main__":
    main()
