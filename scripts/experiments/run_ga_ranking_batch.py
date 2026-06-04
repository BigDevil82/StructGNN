from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunTask:
    algorithm: str
    layout: str
    seed: int
    method: str
    out_path: Path
    cmd: list[str]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run GA FEA-budget comparison experiments.")
    p.add_argument("--layouts", nargs="+", default=["L17_101", "L17_123", "L17_125"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("full", "gnn_cost", "gnn_screen_cost"),
        default=["full", "gnn_cost", "gnn_screen_cost"],
    )
    p.add_argument("--out-dir", default=r"outputs\result\optimization\ranking_batch")
    p.add_argument("--algorithm", choices=("ga", "pso", "optuna", "random"), default="ga")
    p.add_argument("--N", type=int, default=18)
    p.add_argument("--intensity", type=float, default=6.0)
    p.add_argument("--site-class", default="II")
    p.add_argument("--seismic-group", type=int, default=1)
    p.add_argument("--condition-csv", default=None)
    p.add_argument("--steel-price-per-kg", type=float, default=5.0)
    p.add_argument("--ga-pop", type=int, default=8)
    p.add_argument("--ga-gen", type=int, default=3)
    p.add_argument("--ga-elite", type=int, default=1)
    p.add_argument("--ga-mutation", type=float, default=0.3)
    p.add_argument("--pso-swarm", type=int, default=24)
    p.add_argument("--pso-iter", type=int, default=20)
    p.add_argument("--optuna-trials", type=int, default=120)
    p.add_argument("--optuna-startup-trials", type=int, default=24)
    p.add_argument("--optuna-batch-size", type=int, default=24)
    p.add_argument("--random-trials", type=int, default=720)
    p.add_argument("--random-batch-size", type=int, default=48)
    p.add_argument("--optimizer-workers", type=int, default=4)
    p.add_argument("--job-workers", type=int, default=1, help="Number of GA subprocesses to run concurrently.")
    p.add_argument("--eval-ratio", type=float, default=0.5)
    p.add_argument("--min-eval", type=int, default=4)
    p.add_argument("--steel-artifact", default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\gnn_steel.pt")
    p.add_argument("--graph-cache", default=r"data\parametric\cache\gnn_room_graph_cache")
    p.add_argument(
        "--feasibility-artifact",
        default=r"data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt",
    )
    p.add_argument(
        "--feasibility-graph-cache",
        default=r"data\parametric\cache\gnn_room_graph_cache",
    )
    p.add_argument("--feasibility-penalty-cost", type=float, default=1.0e6)
    p.add_argument("--feasibility-hinge-target", type=float, default=0.5)
    p.add_argument("--local-calibration", action="store_true")
    p.add_argument("--local-calibration-min-samples", type=int, default=25)
    p.add_argument("--local-screening-threshold-scale", type=float, default=0.1)
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = _load_conditions(args.condition_csv)
    rows = []
    tasks: list[RunTask] = []
    for layout in args.layouts:
        for seed in args.seeds:
            cond = _condition_for(args, conditions, layout, seed)
            for method in args.methods:
                out_path = out_dir / f"{layout}_seed{seed}_{method}.json"
                cmd = _command(args, layout, seed, method, out_path, cond)
                if args.skip_existing and out_path.exists():
                    print(f"[batch] skip existing {out_path}")
                    rows.append(_summarize(out_path, args.algorithm, layout, seed, method))
                    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
                    continue
                tasks.append(
                    RunTask(
                        algorithm=args.algorithm,
                        layout=layout,
                        seed=seed,
                        method=method,
                        out_path=out_path,
                        cmd=cmd,
                    )
                )

    if args.job_workers <= 1:
        for task in tasks:
            rows.append(_run_task(task, args.continue_on_error, log_dir=None))
            pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    else:
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.job_workers) as executor:
            futures = {
                executor.submit(_run_task, task, args.continue_on_error, log_dir): task for task in tasks
            }
            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    if not args.continue_on_error:
                        raise
                    row = _error_row(task, str(exc))
                rows.append(row)
                completed += 1
                print(f"[batch] finished {completed}/{len(tasks)} {task.out_path.name}")
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
            mean_screened_ratio=("screened_ratio", "mean"),
            mean_cost_preselect_skipped_ratio=("cost_preselect_skipped_ratio", "mean"),
            mean_local_calibration_samples=("local_calibration_samples", "mean"),
        )
        agg.to_csv(out_dir / "summary_by_method.csv")
        print(agg)
        _write_paired_summary(ok, out_dir)


def _run_task(task: RunTask, continue_on_error: bool, log_dir: Path | None) -> dict[str, object]:
    print("[batch]", " ".join(task.cmd))
    try:
        if log_dir is None:
            subprocess.run(task.cmd, cwd=PROJECT_ROOT, check=True)
        else:
            stdout_path = log_dir / f"{task.out_path.stem}.stdout.log"
            stderr_path = log_dir / f"{task.out_path.stem}.stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                subprocess.run(task.cmd, cwd=PROJECT_ROOT, check=True, stdout=stdout, stderr=stderr)
    except subprocess.CalledProcessError as exc:
        if not continue_on_error:
            raise
        return _error_row(task, str(exc))
    return _summarize(task.out_path, task.algorithm, task.layout, task.seed, task.method)


def _error_row(task: RunTask, error: str) -> dict[str, object]:
    return {
        "ok": False,
        "algorithm": task.algorithm,
        "layout_id": task.layout,
        "seed": task.seed,
        "method": task.method,
        "error": error,
        "result_path": str(task.out_path),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def _command(args, layout: str, seed: int, method: str, out_path: Path, cond: dict[str, object]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/shearwall_optimize_main.py",
        "--algorithm",
        str(args.algorithm),
        "--layout-path",
        str(Path("data/dxf/cad_json_data/fem_raw") / f"{layout}.json"),
        "--out",
        str(out_path),
        "--N",
        str(args.N),
        "--intensity",
        str(cond["intensity"]),
        "--site-class",
        str(cond["site_class"]),
        "--seismic-group",
        str(cond["seismic_group"]),
        "--steel-price-per-kg",
        str(args.steel_price_per_kg),
    ]
    if args.algorithm == "ga":
        cmd += [
            "--ga-pop",
            str(args.ga_pop),
            "--ga-gen",
            str(args.ga_gen),
            "--ga-elite",
            str(args.ga_elite),
            "--ga-mutation",
            str(args.ga_mutation),
        ]
    elif args.algorithm == "pso":
        cmd += [
            "--pso-swarm",
            str(args.pso_swarm),
            "--pso-iter",
            str(args.pso_iter),
        ]
    elif args.algorithm == "optuna":
        cmd += [
            "--optuna-trials",
            str(args.optuna_trials),
            "--optuna-startup-trials",
            str(args.optuna_startup_trials),
            "--optuna-batch-size",
            str(args.optuna_batch_size),
        ]
    elif args.algorithm == "random":
        cmd += [
            "--random-trials",
            str(args.random_trials),
            "--random-batch-size",
            str(args.random_batch_size),
        ]
    cmd += [
        "--optimizer-workers",
        str(args.optimizer_workers),
        "--seed",
        str(seed),
    ]
    if method in {"gnn_cost", "gnn_screen_cost"}:
        cmd += [
            "--ga-surrogate-cost-preselect",
            "--ga-cost-eval-ratio",
            str(args.eval_ratio),
            "--ga-cost-min-eval",
            str(args.min_eval),
            "--ga-cost-steel-artifact",
            args.steel_artifact,
            "--ga-cost-graph-cache",
            args.graph_cache,
            "--ga-cost-feasibility-penalty",
            str(args.feasibility_penalty_cost),
            "--ga-cost-feasibility-hinge-target",
            str(args.feasibility_hinge_target),
        ]
        if args.local_calibration:
            cmd += [
                "--ga-local-calibration",
                "--ga-local-calibration-min-samples",
                str(args.local_calibration_min_samples),
                "--ga-local-screening-threshold-scale",
                str(args.local_screening_threshold_scale),
            ]
        if method == "gnn_screen_cost":
            cmd += [
                "--ga-surrogate-screen",
                "--ga-surrogate-artifact",
                args.feasibility_artifact,
                "--ga-surrogate-graph-cache",
                args.feasibility_graph_cache,
            ]
    return cmd


def _load_conditions(path: str | None) -> dict[tuple[str, int], dict[str, object]]:
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"layout_id", "seed", "intensity", "site_class", "seismic_group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Condition CSV missing columns: {sorted(missing)}")
    out = {}
    for row in df.to_dict("records"):
        out[(str(row["layout_id"]), int(row["seed"]))] = {
            "intensity": float(row["intensity"]),
            "site_class": str(row["site_class"]),
            "seismic_group": int(row["seismic_group"]),
        }
    return out


def _condition_for(args, conditions: dict[tuple[str, int], dict[str, object]], layout: str, seed: int) -> dict[str, object]:
    return conditions.get(
        (layout, seed),
        {
            "intensity": float(args.intensity),
            "site_class": str(args.site_class),
            "seismic_group": int(args.seismic_group),
        },
    )


def _summarize(path: Path, algorithm: str, layout: str, seed: int, method: str) -> dict[str, object]:
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
        "algorithm": algorithm,
        "layout_id": layout,
        "seed": seed,
        "method": method,
        "intensity": payload.get("fixed_params", {}).get("intensity"),
        "site_class": payload.get("fixed_params", {}).get("site_class"),
        "seismic_group": payload.get("fixed_params", {}).get("seismic_group"),
        "best_feasible": bool(payload.get("best_feasible", False)),
        "best_objective": float(payload.get("best_objective", float("inf"))),
        "material_cost": float(payload.get("best_objectives", {}).get("material_cost", float("nan"))),
        "fea_calls": int(last.get("final_fea_evaluation_count", last.get("fea_evaluation_count", -1))),
        "first_feasible_fea_calls": first_feasible,
        "screened_ratio": float(last.get("screened_ratio", 0.0)),
        "cost_preselect_skipped_ratio": float(last.get("cost_preselect_skipped_ratio", 0.0)),
        "local_calibration_samples": int(last.get("local_calibration_samples", 0)),
        "local_feasibility_model": last.get("local_feasibility_model", ""),
        "local_steel_model": last.get("local_steel_model", ""),
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
