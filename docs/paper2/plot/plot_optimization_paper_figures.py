from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from optimization_plot_data import (
    DEFAULT_EXPERIMENTS,
    SELECTED15_EXPERIMENTS,
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
    bold_font,
    draw_violin_points,
    family_of,
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
    heatmap_df = load_all_summaries(SELECTED15_EXPERIMENTS)

    plot_combined_distribution_summary(df, paired, args.out_dir)
    plot_tolerance_success_curve(paired, args.out_dir)
    plot_success_heatmap(heatmap_df, args.out_dir)
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
    feasible_cost = df.loc[df["best_feasible"], "material_cost"].dropna()
    cost_ymax = float(feasible_cost.quantile(0.98)) * 1.10 if not feasible_cost.empty else 1.0

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
        data = [
            sub.loc[(sub["method"] == method) & sub["best_feasible"], "material_cost"].dropna().to_numpy()
            for method in METHOD_ORDER
        ]
        draw_violin_points(
            ax,
            data,
            [METHOD_LABELS_SHORT[m] for m in METHOD_ORDER],
            method_colors(METHOD_ORDER),
            ylabel="Best feasible cost" if col_i == 0 else None,
            point_size=5,
        )
        format_million_axis(ax)
        ax.set_ylim(0, cost_ymax)
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
            ylabel="First feasible FEA calls" if col_i == 0 else None,
            point_size=5,
        )
        ax.set_ylim(0, first_ymax)

    for row_i, label in enumerate(["FEA effort", "Cost quality", "Feasible discovery"]):
        axes[row_i, 0].text(
            -0.19,
            0.5,
            label,
            transform=axes[row_i, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontproperties=bold_font(10),
        )
    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    save_figure(fig, "optimization_benchmark_results.png", out_dir)


def plot_tolerance_success_curve(paired: pd.DataFrame, out_dir: str | Path) -> None:
    thresholds = np.array([0, 1, 2, 3, 5, 8, 10, 15], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    curve_colors = {
        "gnn_cost": "#75b7d8",
        "gnn_screen_cost": METHOD_COLORS["gnn_screen_cost"],
    }
    for i, (ax, algorithm) in enumerate(zip(axes, ALGORITHM_ORDER)):
        sub = paired[(paired["algorithm"] == algorithm) & paired["full_feasible"]].copy()
        n_full_feasible = sub[["layout_id", "seed"]].drop_duplicates().shape[0]
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
                color=curve_colors[method],
                label=METHOD_LABELS[method],
            )
            for x, y, count in zip(thresholds, 100 * np.asarray(rates, dtype=float), labels):
                if np.isfinite(y):
                    ax.text(x, y + 1.5, str(count), ha="center", va="bottom", fontsize=6, color="#444444")
        ax.set_title(f"{algorithm} (n={n_full_feasible})")
        ax.set_xlabel("Allowed cost increase vs full (%)")
        ax.set_ylim(0, 105)
        ax.set_xlim(thresholds.min() - 0.6, thresholds.max() + 0.6)
        ax.grid(True)
        if i == 0:
            ax.set_ylabel("Qualified ratio among full-feasible cases (%)")
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=False, labelleft=False)
    axes[-1].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "cost_tolerance_analysis.png", out_dir)


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

    fig, ax = plt.subplots(figsize=(12.5, max(5.0, 0.30 * len(layout_order))))
    cmap = LinearSegmentedColormap.from_list(
        "soft_success",
        ["#75b7d8", "#f7f7f7", "#f47f72"],
    )
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(layout_order)))
    ax.set_yticklabels(layout_display_labels(layout_order))
    ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(layout_order), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.2)
    ax.tick_params(axis="both", left=False, right=False, length=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i, layout in enumerate(layout_order):
        for j, col in enumerate(col_order):
            value = matrix.loc[layout, col]
            if pd.isna(value):
                continue
            text_color = "#ffffff" if value >= 0.78 or value <= 0.18 else "#222222"
            ax.text(
                j,
                i,
                f"{int(counts.loc[layout, col])}/{int(totals.loc[layout, col])}",
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.055, pad=0.20)
    cbar.set_label("Success rate", labelpad=4)
    fig.tight_layout()
    save_figure(fig, "layout_difficulty_heatmap.png", out_dir)


def layout_display_labels(layout_order: list[str]) -> list[str]:
    prefixes = {"L17": "Group7-H1", "L27": "Group7-H2", "L1L28": "Group8"}
    counters = {prefix: 0 for prefix in prefixes}
    labels: list[str] = []
    for layout_id in layout_order:
        prefix = family_of(layout_id)
        group = prefixes.get(prefix, "Group")
        counters[prefix] = counters.get(prefix, 0) + 1
        labels.append(f"{group}-{counters[prefix]}")
    return labels


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
    title_size = 13
    label_size = 11
    tick_size = 10

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
                ax.set_title(METHOD_LABELS[method], fontsize=title_size)
            if col_i == 0:
                ax.set_ylabel("Material cost", fontsize=label_size)
                ax.text(
                    -0.16,
                    0.5,
                    str(case["label"]),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontproperties=bold_font(label_size),
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=False, labelleft=False)
            if row_i == len(selected) - 1:
                ax.set_xlabel("Evaluated candidate", fontsize=label_size)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            ax.tick_params(axis="both", labelsize=tick_size)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.055),
        fontsize=11,
    )
    fig.tight_layout()
    save_figure(fig, "optimization_case_examples.png", out_dir)


def plot_screening_funnel(df: pd.DataFrame, out_dir: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), sharey=True)
    flow_colors = ["#e8a69d", "#efbd75", "#8fb6d6"]
    flow_labels = ["FEA evaluated", "Cost skipped", "Feasibility rejected"]
    for ax, algorithm in zip(axes, ALGORITHM_ORDER):
        rows = df[(df["algorithm"] == algorithm) & (df["method"] == "gnn_screen_cost")]
        flow = aggregate_flow(rows)
        if flow.empty:
            empty_panel(ax, "No flow data")
            continue
        x = flow["step"].to_numpy()
        ax.stackplot(
            x,
            flow["fea"],
            flow["cost_skip"],
            flow["screened"],
            colors=flow_colors,
            labels=flow_labels,
            alpha=0.82,
        )
        ax.plot(x, flow["feasible"], color="#2f7d32", linewidth=1.6, label="FEA feasible")
        ax.set_title(algorithm)
        ax.set_xlabel("Generation / batch")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.margins(x=0)
        ax.set_ylim(0, 1)
        ax.grid(axis="y")
        if ax is axes[0]:
            ax.set_ylabel("Mean candidate fraction")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )
    fig.tight_layout()
    save_figure(fig, "candidate_flow_analysis.png", out_dir)


def select_representative_cases(df: pd.DataFrame, algorithm: str) -> list[dict[str, object]]:
    rows = complete_case_rows(df, algorithm)
    full = rows[rows["method"] == "full"].copy()
    full = add_case_feasibility_stats(full, rows)
    ranked = full[full["feasible_ratio"].notna()].sort_values(
        ["mean_feasible_ratio", "min_feasible_count", "best_feasible", "best_objective", "layout_id", "seed"],
        ascending=[False, False, False, True, True, True],
    )
    selected: list[dict[str, object]] = []
    used_layouts: set[str] = set()

    if not ranked.empty:
        easy = ranked.iloc[0]
        selected.append(case_dict(easy, "Easy"))
        used_layouts.add(str(easy["layout_id"]))

        ratio_min = float(ranked["mean_feasible_ratio"].min())
        ratio_max = float(ranked["mean_feasible_ratio"].max())
        target_ratio = 0.5 * (ratio_min + ratio_max)
        medium_pool = ranked[ranked["min_feasible_count"] >= 50].copy()
        if medium_pool.empty:
            medium_pool = ranked[ranked["min_feasible_count"] > 0].copy()
        if medium_pool.empty:
            medium_pool = ranked
        medium_ranked = medium_pool.assign(
            _target_dist=(medium_pool["mean_feasible_ratio"] - target_ratio).abs()
        ).sort_values(
            ["_target_dist", "min_feasible_count", "layout_id", "seed"],
            ascending=[True, False, True, True],
        )
        medium = _pick_distinct_case(medium_ranked, 0, used_layouts)
        if medium is not None:
            selected.append(case_dict(medium, "Medium"))
            used_layouts.add(str(medium["layout_id"]))

        hard_ranked = ranked.sort_values(
            [
                "mean_feasible_ratio",
                "min_feasible_count",
                "best_feasible",
                "best_objective",
                "layout_id",
                "seed",
            ],
            ascending=[True, True, True, True, True, True],
        )
        hard = _pick_distinct_case(hard_ranked, 0, used_layouts)
        if hard is not None:
            selected.append(case_dict(hard, "Hard"))
    return selected[:3]


def add_case_feasibility_stats(full: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    full = full.copy()
    full_ratios: list[float] = []
    mean_ratios: list[float] = []
    min_counts: list[int] = []
    mean_counts: list[float] = []
    for _, row in full.iterrows():
        case_rows = rows[(rows["layout_id"] == row["layout_id"]) & (rows["seed"] == row["seed"])]
        ratios: list[float] = []
        counts: list[int] = []
        for method in METHOD_ORDER:
            method_rows = case_rows[case_rows["method"] == method]
            if method_rows.empty:
                continue
            ratio, count = evaluated_feasible_stats(method_rows.iloc[0])
            if np.isfinite(ratio):
                ratios.append(ratio)
                counts.append(count)
        full_ratio, _ = evaluated_feasible_stats(row)
        full_ratios.append(full_ratio)
        mean_ratios.append(float(np.mean(ratios)) if ratios else float("nan"))
        min_counts.append(int(min(counts)) if counts else 0)
        mean_counts.append(float(np.mean(counts)) if counts else 0.0)
    full["feasible_ratio"] = full_ratios
    full["mean_feasible_ratio"] = mean_ratios
    full["min_feasible_count"] = min_counts
    full["mean_feasible_count"] = mean_counts
    return full


def evaluated_feasible_stats(row: pd.Series) -> tuple[float, int]:
    path = find_result_json(row)
    if path is None:
        return float("nan"), 0
    points = material_points(read_payload(path).get("history", []))
    if points.empty:
        return float("nan"), 0
    return float(points["feasible"].mean()), int(points["feasible"].sum())


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
        s=12,
        color="#b9b9b9",
        edgecolor="none",
        alpha=0.50,
        label="Infeasible",
    )
    ax.scatter(
        feasible["evaluation"],
        feasible["material_cost"],
        s=12,
        color="#4e9f65",
        edgecolor="#1f5c35",
        linewidth=0.25,
        alpha=0.82,
        label="Feasible",
    )
    line = points.dropna(subset=["best_feasible_cost"])
    if not line.empty:
        ax.plot(
            line["evaluation"],
            line["best_feasible_cost"],
            color="#efbd75",
            linewidth=1.8,
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
