from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SITE_CLASSES = ["I0", "I", "II", "III", "IV"]
SEISMIC_GROUPS = [1, 2, 3]
INTENSITY_BY_FAMILY = {
    "L17": 7.0,
    "L27": 7.5,
    "L1L28": 8.0,
}
DEFAULT_SEEDS = [42, 7, 2025, 3407, 1009]
TEST_LAYOUTS = [
    "L17_123",
    "L17_125",
    "L17_127",
    "L17_157",
    "L17_159",
    "L17_223",
    "L17_226",
    "L1L28_11",
    "L1L28_25",
    "L1L28_30",
    "L1L28_44",
    "L1L28_47",
    "L1L28_74",
    "L1L28_80",
    "L1L28_198",
    "L1L28_206",
    "L1L28_215",
    "L1L28_221",
    "L27_57",
    "L27_59",
    "L27_100",
    "L27_151",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run primary surrogate-assisted optimization experiment on test layouts.")
    p.add_argument("--out-root", default=r"outputs\result\optimization\ranking_test_primary")
    p.add_argument("--algorithm", choices=["ga", "pso", "optuna", "random"], default="ga")
    p.add_argument("--layout-source", choices=["builtin", "split"], default="builtin")
    p.add_argument(
        "--split-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    p.add_argument("--layout-json-dir", default=r"data\dxf\cad_json_data\fem_raw")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--condition-seed", type=int, default=20260603)
    p.add_argument("--ga-pop", type=int, default=24)
    p.add_argument("--ga-gen", type=int, default=8)
    p.add_argument("--ga-elite", type=int, default=2)
    p.add_argument("--pso-swarm", type=int, default=24)
    p.add_argument("--pso-iter", type=int, default=8)
    p.add_argument("--pso-mutation-rate", type=float, default=0.0)
    p.add_argument("--optuna-trials", type=int, default=360)
    p.add_argument("--optuna-startup-trials", type=int, default=48)
    p.add_argument("--optuna-batch-size", type=int, default=24)
    p.add_argument("--random-trials", type=int, default=720)
    p.add_argument("--random-batch-size", type=int, default=48)
    p.add_argument("--optimizer-workers", type=int, default=8)
    p.add_argument("--job-workers", type=int, default=1)
    p.add_argument("--eval-ratio", type=float, default=0.5)
    p.add_argument("--min-eval", type=int, default=8)
    p.add_argument("--feasibility-penalty-cost", type=float, default=1.0e6)
    p.add_argument("--feasibility-hinge-target", type=float, default=0.5)
    p.add_argument("--local-calibration", action="store_true")
    p.add_argument("--skip-full", action="store_true", help="Run only surrogate-assisted methods.")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_dir = out_root / "primary_large"
    out_dir.mkdir(parents=True, exist_ok=True)

    layouts = _load_test_layouts(args.layout_source, Path(args.split_path), Path(args.layout_json_dir))
    conditions = _make_conditions(layouts, args.seeds, args.condition_seed)
    layout_path = out_root / "test_layouts.csv"
    condition_path = out_root / "conditions.csv"
    pd.DataFrame({"layout_id": layouts, "family": [_family(x) for x in layouts]}).to_csv(layout_path, index=False)
    pd.DataFrame(conditions).to_csv(condition_path, index=False)

    methods = ["gnn_cost", "gnn_screen_cost"] if args.skip_full else ["full", "gnn_cost", "gnn_screen_cost"]
    cmd = [
        sys.executable,
        "scripts/experiments/run_ga_ranking_batch.py",
        "--algorithm",
        args.algorithm,
        "--layouts",
        *layouts,
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--methods",
        *methods,
        "--out-dir",
        str(out_dir),
        "--condition-csv",
        str(condition_path),
        "--optimizer-workers",
        str(args.optimizer_workers),
        "--job-workers",
        str(args.job_workers),
        "--eval-ratio",
        str(args.eval_ratio),
        "--min-eval",
        str(args.min_eval),
        "--feasibility-penalty-cost",
        str(args.feasibility_penalty_cost),
        "--feasibility-hinge-target",
        str(args.feasibility_hinge_target),
        "--continue-on-error",
        "--skip-existing",
    ]
    if args.algorithm == "ga":
        cmd += [
            "--ga-pop",
            str(args.ga_pop),
            "--ga-gen",
            str(args.ga_gen),
            "--ga-elite",
            str(args.ga_elite),
        ]
    elif args.algorithm == "pso":
        cmd += [
            "--pso-swarm",
            str(args.pso_swarm),
            "--pso-iter",
            str(args.pso_iter),
            "--pso-mutation-rate",
            str(args.pso_mutation_rate),
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
    if args.local_calibration:
        cmd.append("--local-calibration")
    print("[test-primary]", " ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        _write_combined_summary(out_root)


def _load_test_layouts(source: str, split_path: Path, layout_json_dir: Path) -> list[str]:
    if source == "builtin":
        return [layout for layout in TEST_LAYOUTS if (layout_json_dir / f"{layout}.json").exists()]

    df = pd.read_parquet(split_path, columns=["layout_id", "split"])
    layouts = sorted(
        df.loc[df["split"].astype(str).eq("test"), "layout_id"].astype(str).unique(),
        key=_layout_sort_key,
    )
    layouts = [layout for layout in layouts if (layout_json_dir / f"{layout}.json").exists()]
    allowed = {"L17", "L27", "L1L28"}
    return [layout for layout in layouts if _family(layout) in allowed]


def _make_conditions(layouts: list[str], seeds: list[int], condition_seed: int) -> list[dict[str, object]]:
    rows = []
    for layout in layouts:
        family = _family(layout)
        intensity = INTENSITY_BY_FAMILY[family]
        for seed in seeds:
            rng = random.Random(f"{condition_seed}:{layout}:{seed}")
            rows.append(
                {
                    "layout_id": layout,
                    "family": family,
                    "seed": int(seed),
                    "intensity": intensity,
                    "site_class": rng.choice(SITE_CLASSES),
                    "seismic_group": rng.choice(SEISMIC_GROUPS),
                }
            )
    return rows


def _family(layout: str) -> str:
    if layout.startswith("L1L28_"):
        return "L1L28"
    return layout.split("_", 1)[0]


def _layout_sort_key(layout: str) -> tuple[str, int]:
    family = _family(layout)
    try:
        number = int(layout.rsplit("_", 1)[1])
    except ValueError:
        number = 10**9
    return family, number


def _write_combined_summary(out_root: Path) -> None:
    rows = []
    for path in sorted(out_root.glob("*/summary_by_method.csv")):
        df = pd.read_csv(path)
        df.insert(0, "experiment", path.parent.name)
        rows.append(df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_root / "combined_summary_by_method.csv", index=False)

    paired_rows = []
    for path in sorted(out_root.glob("*/paired_summary_by_method.csv")):
        df = pd.read_csv(path)
        df.insert(0, "experiment", path.parent.name)
        paired_rows.append(df)
    if paired_rows:
        pd.concat(paired_rows, ignore_index=True).to_csv(out_root / "combined_paired_summary_by_method.csv", index=False)


if __name__ == "__main__":
    main()
