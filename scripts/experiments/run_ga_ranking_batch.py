from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GA FEA-budget comparison experiments.")
    p.add_argument("--layouts", nargs="+", default=["L17_101", "L17_123", "L17_125"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    p.add_argument("--methods", nargs="+", choices=("full", "random", "gnn_rank"), default=["full", "random", "gnn_rank"])
    p.add_argument("--out-dir", default=r"outputs\result\optimization\ranking_batch")
    p.add_argument("--N", type=int, default=18)
    p.add_argument("--intensity", type=float, default=6.0)
    p.add_argument("--site-class", default="II")
    p.add_argument("--seismic-group", type=int, default=1)
    p.add_argument("--ga-pop", type=int, default=8)
    p.add_argument("--ga-gen", type=int, default=3)
    p.add_argument("--ga-elite", type=int, default=1)
    p.add_argument("--ga-mutation", type=float, default=0.3)
    p.add_argument("--optimizer-workers", type=int, default=4)
    p.add_argument("--eval-ratio", type=float, default=0.5)
    p.add_argument("--min-eval", type=int, default=4)
    p.add_argument("--random-ratio", type=float, default=0.125)
    p.add_argument("--steel-ranking-artifact", default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\gnn_steel.pt")
    p.add_argument("--steel-ranking-graph-cache", default=r"data\parametric\cache\gnn_room_graph_cache")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for layout in args.layouts:
        for seed in args.seeds:
            for method in args.methods:
                out_path = out_dir / f"{layout}_seed{seed}_{method}.json"
                cmd = _command(args, layout, seed, method, out_path)
                if args.skip_existing and out_path.exists():
                    print(f"[batch] skip existing {out_path}")
                    rows.append(_summarize(out_path, layout, seed, method))
                    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
                    continue
                print("[batch]", " ".join(cmd))
                try:
                    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append(
                        {
                            "ok": False,
                            "layout_id": layout,
                            "seed": seed,
                            "method": method,
                            "error": str(exc),
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    continue
                rows.append(_summarize(out_path, layout, seed, method))
                pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if not summary.empty and "ok" in summary.columns:
        ok = summary[summary["ok"] == True].copy()
    else:
        ok = summary
    if not ok.empty:
        agg = ok.groupby("method").agg(
            runs=("method", "size"),
            feasible_rate=("best_feasible", "mean"),
            mean_best_objective=("best_objective", "mean"),
            mean_material_cost=("material_cost", "mean"),
            mean_fea_calls=("fea_calls", "mean"),
            median_fea_calls=("fea_calls", "median"),
            mean_steel_rank_skipped_ratio=("steel_rank_skipped_ratio", "mean"),
            mean_random_skipped_ratio=("random_preselect_skipped_ratio", "mean"),
        )
        agg.to_csv(out_dir / "summary_by_method.csv")
        print(agg)
        _write_paired_summary(ok, out_dir)


def _command(args, layout: str, seed: int, method: str, out_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/shearwall_optimize_main.py",
        "--algorithm",
        "ga",
        "--layout-path",
        str(Path("data/dxf/cad_json_data/fem_raw") / f"{layout}.json"),
        "--out",
        str(out_path),
        "--N",
        str(args.N),
        "--intensity",
        str(args.intensity),
        "--site-class",
        args.site_class,
        "--seismic-group",
        str(args.seismic_group),
        "--ga-pop",
        str(args.ga_pop),
        "--ga-gen",
        str(args.ga_gen),
        "--ga-elite",
        str(args.ga_elite),
        "--ga-mutation",
        str(args.ga_mutation),
        "--optimizer-workers",
        str(args.optimizer_workers),
        "--seed",
        str(seed),
    ]
    if method == "random":
        cmd += [
            "--ga-random-preselect",
            "--ga-random-preselect-eval-ratio",
            str(args.eval_ratio),
            "--ga-random-preselect-min-eval",
            str(args.min_eval),
        ]
    elif method == "gnn_rank":
        cmd += [
            "--ga-steel-ranking",
            "--ga-steel-ranking-eval-ratio",
            str(args.eval_ratio),
            "--ga-steel-ranking-min-eval",
            str(args.min_eval),
            "--ga-steel-ranking-random-ratio",
            str(args.random_ratio),
            "--ga-steel-ranking-artifact",
            args.steel_ranking_artifact,
            "--ga-steel-ranking-graph-cache",
            args.steel_ranking_graph_cache,
        ]
    return cmd


def _summarize(path: Path, layout: str, seed: int, method: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    last = history[-1] if history else {}
    first_feasible = None
    for item in history:
        if item.get("feasible_count", 0) > 0:
            first_feasible = item.get("fea_evaluation_count")
            break
    return {
        "ok": True,
        "layout_id": layout,
        "seed": seed,
        "method": method,
        "best_feasible": bool(payload.get("best_feasible", False)),
        "best_objective": float(payload.get("best_objective", float("inf"))),
        "material_cost": float(payload.get("best_objectives", {}).get("material_cost", float("nan"))),
        "fea_calls": int(last.get("final_fea_evaluation_count", last.get("fea_evaluation_count", -1))),
        "first_feasible_fea_calls": first_feasible,
        "steel_rank_skipped_ratio": float(last.get("steel_rank_skipped_ratio", 0.0)),
        "random_preselect_skipped_ratio": float(last.get("random_preselect_skipped_ratio", 0.0)),
        "result_path": str(path),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _write_paired_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if "full" not in set(summary["method"]):
        return
    rows = []
    full = summary[summary["method"] == "full"].set_index(["layout_id", "seed"])
    for _, row in summary[summary["method"] != "full"].iterrows():
        key = (row["layout_id"], row["seed"])
        if key not in full.index:
            continue
        base = full.loc[key]
        base_obj = float(base["best_objective"])
        obj = float(row["best_objective"])
        base_calls = float(base["fea_calls"])
        calls = float(row["fea_calls"])
        rows.append(
            {
                "layout_id": row["layout_id"],
                "seed": int(row["seed"]),
                "method": row["method"],
                "full_feasible": bool(base["best_feasible"]),
                "method_feasible": bool(row["best_feasible"]),
                "full_objective": base_obj,
                "method_objective": obj,
                "objective_delta": obj - base_obj,
                "objective_ratio": obj / base_obj if base_obj > 0 else float("nan"),
                "full_fea_calls": base_calls,
                "method_fea_calls": calls,
                "fea_reduction": (base_calls - calls) / base_calls if base_calls > 0 else float("nan"),
            }
        )
    paired = pd.DataFrame(rows)
    if paired.empty:
        return
    paired.to_csv(out_dir / "paired_summary.csv", index=False)
    agg = paired.groupby("method").agg(
        pairs=("method", "size"),
        method_feasible_rate=("method_feasible", "mean"),
        full_feasible_rate=("full_feasible", "mean"),
        mean_objective_delta=("objective_delta", "mean"),
        mean_objective_ratio=("objective_ratio", "mean"),
        mean_fea_reduction=("fea_reduction", "mean"),
        median_fea_reduction=("fea_reduction", "median"),
    )
    agg.to_csv(out_dir / "paired_summary_by_method.csv")
    print(agg)


if __name__ == "__main__":
    main()
