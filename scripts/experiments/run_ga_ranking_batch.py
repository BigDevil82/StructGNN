from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
                print("[batch]", " ".join(cmd))
                try:
                    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
                except subprocess.CalledProcessError as exc:
                    if not args.continue_on_error:
                        raise
                    rows.append({"layout_id": layout, "seed": seed, "method": method, "ok": False, "error": str(exc)})
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
    }


if __name__ == "__main__":
    main()
