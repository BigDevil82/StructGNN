from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    commands = []
    if not args.skip_feasibility:
        commands.append(("feasibility", feasibility_command(args, out_root / "feasibility")))
    if not args.skip_steel:
        commands.append(("steel", steel_command(args, out_root / "steel")))

    manifest = {
        "out_root": str(out_root),
        "commands": [{"name": name, "command": cmd} for name, cmd in commands],
    }
    (out_root / "commands.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    if args.dry_run:
        for name, cmd in commands:
            print(f"[paper2][local-calib][dry-run] {name}: {' '.join(cmd)}")
        return

    for name, cmd in commands:
        print(f"[paper2][local-calib] running {name}")
        subprocess.run(cmd, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate paper Section 3.3 local calibration experiment data.")
    p.add_argument(
        "--dataset-path",
        default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet",
    )
    p.add_argument(
        "--feasibility-prediction-path",
        default=r"data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1\predictions_test.parquet",
    )
    p.add_argument(
        "--steel-prediction-path",
        default=r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\predictions_test.parquet",
    )
    p.add_argument("--out-root", default=r"outputs\result\paper2\local_calibration")
    p.add_argument("--layouts", default="all")
    p.add_argument("--max-layouts", type=int, default=0)
    p.add_argument("--calib-sizes", default="25,50,100,200,500")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-eval-samples", type=int, default=200)
    p.add_argument(
        "--feasibility-methods",
        default="platt,isotonic,logistic,residual_lgbm",
        help="Subset of methods supported by scripts/local_calibration/feasibility_probability_calibration.py.",
    )
    p.add_argument("--feasibility-input-mode", choices=["prob", "param", "prob_param"], default="prob_param")
    p.add_argument("--screening-target-recall", type=float, default=0.995)
    p.add_argument("--screening-threshold-scale", type=float, default=1.0)
    p.add_argument(
        "--steel-models",
        default="ridge,gpr,lgbm,knn",
        help="Subset of models supported by scripts/local_calibration/steel_residual_calibration.py.",
    )
    p.add_argument("--steel-input-mode", choices=["pred", "param", "pred_param"], default="pred_param")
    p.add_argument("--steel-target-space", choices=["kg", "log"], default="kg")
    p.add_argument("--max-gpr-train", type=int, default=300)
    p.add_argument("--skip-feasibility", action="store_true")
    p.add_argument("--skip-steel", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def feasibility_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/local_calibration/feasibility_probability_calibration.py",
        "--dataset-path",
        args.dataset_path,
        "--prediction-path",
        args.feasibility_prediction_path,
        "--output-dir",
        str(out_dir),
        "--layouts",
        args.layouts,
        "--max-layouts",
        str(args.max_layouts),
        "--calib-sizes",
        args.calib_sizes,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--min-eval-samples",
        str(args.min_eval_samples),
        "--methods",
        args.feasibility_methods,
        "--input-mode",
        args.feasibility_input_mode,
        "--screening-target-recall",
        str(args.screening_target_recall),
        "--screening-threshold-scale",
        str(args.screening_threshold_scale),
    ]


def steel_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/local_calibration/steel_residual_calibration.py",
        "--dataset-path",
        args.dataset_path,
        "--prediction-path",
        args.steel_prediction_path,
        "--output-dir",
        str(out_dir),
        "--layouts",
        args.layouts,
        "--max-layouts",
        str(args.max_layouts),
        "--calib-sizes",
        args.calib_sizes,
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--min-eval-samples",
        str(args.min_eval_samples),
        "--models",
        args.steel_models,
        "--input-mode",
        args.steel_input_mode,
        "--target-space",
        args.steel_target_space,
        "--max-gpr-train",
        str(args.max_gpr_train),
    ]


if __name__ == "__main__":
    main()
