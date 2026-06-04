from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze GA optimization batch results.")
    p.add_argument("--root", default=r"outputs\result\optimization\ranking_test_primary_pop48_gen15_parallel")
    p.add_argument("--summary", default=None, help="Optional direct path to summary.csv.")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--baseline-method", default="full")
    return p


def main() -> None:
    args = build_parser().parse_args()
    summary_path = _summary_path(args)
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    df = _prepare(df)
    baseline = str(args.baseline_method)

    method = _method_summary(df)
    family = _family_method_summary(df)
    layout = _layout_method_summary(df)
    pair = _paired(df, baseline)
    pair_method = _paired_method_summary(pair)
    transition = _transition_summary(pair)
    first = _first_feasible_summary(df)
    hard = _hard_layout_summary(df, baseline)
    misses = _missed_feasible_cases(pair)
    wins = _method_only_feasible_cases(pair)

    tables = {
        "method_summary.csv": method,
        "family_method_summary.csv": family,
        "layout_method_summary.csv": layout,
        "paired_cases.csv": pair,
        "paired_method_summary.csv": pair_method,
        "feasibility_transition_summary.csv": transition,
        "first_feasible_summary.csv": first,
        "hard_layout_summary.csv": hard,
        "missed_feasible_cases.csv": misses,
        "method_only_feasible_cases.csv": wins,
    }
    for name, table in tables.items():
        table.to_csv(out_dir / name, index=False)

    report = _report(
        summary_path=summary_path,
        out_dir=out_dir,
        df=df,
        method=method,
        family=family,
        pair_method=pair_method,
        transition=transition,
        hard=hard,
        misses=misses,
        wins=wins,
        baseline=baseline,
    )
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[analysis] wrote {out_dir}")


def _summary_path(args: argparse.Namespace) -> Path:
    if args.summary:
        return Path(args.summary)
    root = Path(args.root)
    direct = root / "summary.csv"
    if direct.exists():
        return direct
    primary = root / "primary_large" / "summary.csv"
    if primary.exists():
        return primary
    matches = sorted(root.glob("*/summary.csv"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find a unique summary.csv under {root}.")


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df["layout_id"].astype(str).map(_family)
    for col in [
        "best_feasible",
        "best_objective",
        "material_cost",
        "fea_calls",
        "first_feasible_fea_calls",
        "screened_ratio",
        "cost_preselect_skipped_ratio",
        "local_calibration_samples",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "best_feasible" in df:
        df["best_feasible"] = df["best_feasible"].astype(bool)
    return df


def _family(layout_id: str) -> str:
    if layout_id.startswith("L1L28_"):
        return "L1L28"
    return layout_id.split("_", 1)[0]


def _method_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", dropna=False)
        .agg(
            runs=("method", "size"),
            feasible_rate=("best_feasible", "mean"),
            feasible_count=("best_feasible", "sum"),
            mean_best_objective=("best_objective", "mean"),
            median_best_objective=("best_objective", "median"),
            mean_material_cost=("material_cost", "mean"),
            median_material_cost=("material_cost", "median"),
            mean_fea_calls=("fea_calls", "mean"),
            median_fea_calls=("fea_calls", "median"),
            mean_screened_ratio=("screened_ratio", "mean"),
            mean_cost_preselect_skipped_ratio=("cost_preselect_skipped_ratio", "mean"),
            mean_local_calibration_samples=("local_calibration_samples", "mean"),
        )
        .reset_index()
    )


def _family_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["family", "method"], dropna=False)
        .agg(
            runs=("method", "size"),
            feasible_rate=("best_feasible", "mean"),
            feasible_count=("best_feasible", "sum"),
            mean_best_objective=("best_objective", "mean"),
            mean_material_cost=("material_cost", "mean"),
            mean_fea_calls=("fea_calls", "mean"),
            median_fea_calls=("fea_calls", "median"),
            mean_screened_ratio=("screened_ratio", "mean"),
            mean_cost_preselect_skipped_ratio=("cost_preselect_skipped_ratio", "mean"),
        )
        .reset_index()
    )


def _layout_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["family", "layout_id", "method"], dropna=False)
        .agg(
            runs=("method", "size"),
            feasible_count=("best_feasible", "sum"),
            feasible_rate=("best_feasible", "mean"),
            mean_best_objective=("best_objective", "mean"),
            mean_material_cost=("material_cost", "mean"),
            mean_fea_calls=("fea_calls", "mean"),
        )
        .reset_index()
        .sort_values(["family", "layout_id", "method"])
    )


def _paired(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base = df[df["method"].eq(baseline)].set_index(["layout_id", "seed"])
    rows: list[dict[str, object]] = []
    for _, row in df[~df["method"].eq(baseline)].iterrows():
        key = (row["layout_id"], row["seed"])
        if key not in base.index:
            continue
        b = base.loc[key]
        base_obj = float(b["best_objective"])
        obj = float(row["best_objective"])
        base_mat = float(b["material_cost"])
        mat = float(row["material_cost"])
        base_fea = float(b["fea_calls"])
        fea = float(row["fea_calls"])
        rows.append(
            {
                "layout_id": row["layout_id"],
                "seed": int(row["seed"]),
                "family": row["family"],
                "method": row["method"],
                "baseline_feasible": bool(b["best_feasible"]),
                "method_feasible": bool(row["best_feasible"]),
                "baseline_objective": base_obj,
                "method_objective": obj,
                "objective_delta": obj - base_obj,
                "objective_ratio": obj / base_obj if base_obj > 0 else np.nan,
                "baseline_material_cost": base_mat,
                "method_material_cost": mat,
                "material_cost_delta": mat - base_mat,
                "material_cost_ratio": mat / base_mat if base_mat > 0 else np.nan,
                "baseline_fea_calls": base_fea,
                "method_fea_calls": fea,
                "fea_reduction": (base_fea - fea) / base_fea if base_fea > 0 else np.nan,
                "baseline_first_feasible_fea_calls": b.get("first_feasible_fea_calls", np.nan),
                "method_first_feasible_fea_calls": row.get("first_feasible_fea_calls", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _paired_method_summary(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pair
    rows = []
    for method, g in pair.groupby("method", dropna=False):
        both = g[g["baseline_feasible"] & g["method_feasible"]]
        rows.append(
            {
                "method": method,
                "pairs": len(g),
                "baseline_feasible_rate": g["baseline_feasible"].mean(),
                "method_feasible_rate": g["method_feasible"].mean(),
                "both_feasible_count": len(both),
                "baseline_only_feasible_count": int((g["baseline_feasible"] & ~g["method_feasible"]).sum()),
                "method_only_feasible_count": int((~g["baseline_feasible"] & g["method_feasible"]).sum()),
                "neither_feasible_count": int((~g["baseline_feasible"] & ~g["method_feasible"]).sum()),
                "mean_fea_reduction": g["fea_reduction"].mean(),
                "median_fea_reduction": g["fea_reduction"].median(),
                "both_feasible_mean_objective_ratio": both["objective_ratio"].mean(),
                "both_feasible_median_objective_ratio": both["objective_ratio"].median(),
                "both_feasible_p90_objective_ratio": both["objective_ratio"].quantile(0.9),
                "both_feasible_better_or_equal_count": int((both["method_objective"] <= both["baseline_objective"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _transition_summary(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pair
    rows = []
    for method, g in pair.groupby("method", dropna=False):
        states = {
            "both_feasible": g["baseline_feasible"] & g["method_feasible"],
            "baseline_only_feasible": g["baseline_feasible"] & ~g["method_feasible"],
            "method_only_feasible": ~g["baseline_feasible"] & g["method_feasible"],
            "neither_feasible": ~g["baseline_feasible"] & ~g["method_feasible"],
        }
        for state, mask in states.items():
            part = g[mask]
            rows.append(
                {
                    "method": method,
                    "state": state,
                    "count": len(part),
                    "mean_fea_reduction": part["fea_reduction"].mean(),
                    "mean_objective_ratio": part["objective_ratio"].mean(),
                    "median_objective_ratio": part["objective_ratio"].median(),
                }
            )
    return pd.DataFrame(rows)


def _first_feasible_summary(df: pd.DataFrame) -> pd.DataFrame:
    found = df[df["first_feasible_fea_calls"].notna()].copy()
    if found.empty:
        return pd.DataFrame()
    return (
        found.groupby("method", dropna=False)
        .agg(
            found_count=("first_feasible_fea_calls", "size"),
            mean_first_feasible_fea_calls=("first_feasible_fea_calls", "mean"),
            median_first_feasible_fea_calls=("first_feasible_fea_calls", "median"),
            p90_first_feasible_fea_calls=("first_feasible_fea_calls", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )


def _hard_layout_summary(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base = df[df["method"].eq(baseline)]
    return (
        base.groupby(["family", "layout_id"], dropna=False)
        .agg(
            runs=("method", "size"),
            baseline_feasible_count=("best_feasible", "sum"),
            baseline_feasible_rate=("best_feasible", "mean"),
            mean_baseline_objective=("best_objective", "mean"),
            mean_baseline_fea_calls=("fea_calls", "mean"),
        )
        .reset_index()
        .sort_values(["baseline_feasible_count", "family", "layout_id"])
    )


def _missed_feasible_cases(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pair
    return pair[pair["baseline_feasible"] & ~pair["method_feasible"]].sort_values(
        ["method", "objective_ratio"], ascending=[True, False]
    )


def _method_only_feasible_cases(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pair
    return pair[~pair["baseline_feasible"] & pair["method_feasible"]].sort_values(
        ["method", "objective_ratio"], ascending=[True, True]
    )


def _report(
    *,
    summary_path: Path,
    out_dir: Path,
    df: pd.DataFrame,
    method: pd.DataFrame,
    family: pd.DataFrame,
    pair_method: pd.DataFrame,
    transition: pd.DataFrame,
    hard: pd.DataFrame,
    misses: pd.DataFrame,
    wins: pd.DataFrame,
    baseline: str,
) -> str:
    lines = [
        "# GA Optimization Result Analysis",
        "",
        f"- summary: `{summary_path}`",
        f"- output: `{out_dir}`",
        f"- rows: {len(df)}",
        f"- layouts: {df['layout_id'].nunique()}",
        f"- seeds: {df['seed'].nunique()}",
        f"- baseline: `{baseline}`",
        "",
        "## Method Summary",
        "",
        _md(method),
        "",
        "## Paired Method Summary",
        "",
        _md(pair_method),
        "",
        "## Family Summary",
        "",
        _md(family),
        "",
        "## Feasibility Transitions",
        "",
        _md(transition),
        "",
        "## Hard Layouts By Baseline",
        "",
        _md(hard.head(12)),
        "",
        "## Missed Feasible Cases",
        "",
        _md(misses.head(12)),
        "",
        "## Method-Only Feasible Cases",
        "",
        _md(wins.head(12)),
        "",
    ]
    return "\n".join(lines)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    out = out.astype(str)
    cols = list(out.columns)
    rows = [cols, ["---"] * len(cols)]
    rows.extend(out[cols].values.tolist())
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(cols))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)) + " |"

    return "\n".join(fmt(row) for row in rows)


if __name__ == "__main__":
    main()
