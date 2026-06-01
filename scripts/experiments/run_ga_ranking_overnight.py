from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LARGE_LAYOUTS = [
    "L17_101",
    "L17_102",
    "L17_109",
    "L17_123",
    "L17_125",
    "L17_130",
    "L17_137",
    "L17_145",
    "L17_155",
    "L17_208",
]
SMALL_LAYOUTS = ["L17_101", "L17_123", "L17_145", "L17_208"]
LARGE_SEEDS = [42, 7, 2025, 3407]
SMALL_SEEDS = [42, 7]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run overnight GA ranking experiments.")
    p.add_argument("--out-root", default=r"outputs\result\optimization\ranking_overnight")
    p.add_argument("--only", nargs="+", choices=("primary_large", "ratio_025", "ratio_075", "robust_i7"))
    p.add_argument("--ga-pop", type=int, default=16)
    p.add_argument("--ga-gen", type=int, default=8)
    p.add_argument("--optimizer-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    experiments = _experiments(args, out_root)
    if args.only:
        selected = set(args.only)
        experiments = [item for item in experiments if item[0] in selected]

    for name, cmd in experiments:
        print(f"[overnight] {name}")
        print("[overnight]", " ".join(cmd))
        if args.dry_run:
            continue
        t0 = time.time()
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        print(f"[overnight] finished {name} in {(time.time() - t0) / 60.0:.1f} min")

    if not args.dry_run:
        _write_combined_summary(out_root)


def _experiments(args, out_root: Path) -> list[tuple[str, list[str]]]:
    common = [
        "--ga-pop",
        str(args.ga_pop),
        "--ga-gen",
        str(args.ga_gen),
        "--ga-elite",
        "2",
        "--optimizer-workers",
        str(args.optimizer_workers),
        "--continue-on-error",
        "--skip-existing",
    ]
    return [
        (
            "primary_large",
            _batch_cmd(
                out_root / "primary_large",
                layouts=LARGE_LAYOUTS,
                seeds=LARGE_SEEDS,
                eval_ratio=0.5,
                min_eval=8,
                random_ratio=0.125,
                extra=common,
            ),
        ),
        (
            "ratio_025",
            _batch_cmd(
                out_root / "ratio_025",
                layouts=SMALL_LAYOUTS,
                seeds=SMALL_SEEDS,
                eval_ratio=0.25,
                min_eval=4,
                random_ratio=0.125,
                extra=common,
            ),
        ),
        (
            "ratio_075",
            _batch_cmd(
                out_root / "ratio_075",
                layouts=SMALL_LAYOUTS,
                seeds=SMALL_SEEDS,
                eval_ratio=0.75,
                min_eval=8,
                random_ratio=0.125,
                extra=common,
            ),
        ),
        (
            "robust_i7",
            _batch_cmd(
                out_root / "robust_i7",
                layouts=SMALL_LAYOUTS,
                seeds=SMALL_SEEDS,
                eval_ratio=0.5,
                min_eval=8,
                random_ratio=0.125,
                extra=common + ["--intensity", "7.0"],
            ),
        ),
    ]


def _batch_cmd(
    out_dir: Path,
    *,
    layouts: list[str],
    seeds: list[int],
    eval_ratio: float,
    min_eval: int,
    random_ratio: float,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable,
        "scripts/experiments/run_ga_ranking_batch.py",
        "--layouts",
        *layouts,
        "--seeds",
        *[str(seed) for seed in seeds],
        "--methods",
        "full",
        "random",
        "gnn_rank",
        "--out-dir",
        str(out_dir),
        "--eval-ratio",
        str(eval_ratio),
        "--min-eval",
        str(min_eval),
        "--random-ratio",
        str(random_ratio),
        *extra,
    ]


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
