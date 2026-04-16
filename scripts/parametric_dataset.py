import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.shearwall_modeling import DatasetGenerationConfig, generate_structural_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate parametric structural dataset for all layouts.")
    p.add_argument("--layout-dir", default=r"data\dxf\cad_json_data\fem_raw")
    p.add_argument("--output-dir", default=r"outputs\result\parametric_dataset")
    p.add_argument("--samples-per-layout", type=int, default=500)
    p.add_argument("--samples-per-task", type=int, default=20)
    p.add_argument("--progress-log-interval", type=int, default=200)
    p.add_argument("--sampling-method", choices=["random", "lhs"], default="lhs")
    p.add_argument("--storage-format", choices=["parquet", "hdf5"], default="parquet")
    p.add_argument(
        "--builder", choices=["mvlem_frame", "equivalent_frame", "detailed_shell"], default="mvlem_frame"
    )
    p.add_argument("--num-modes", type=int, default=6)
    p.add_argument("--max-workers", type=int, default=None, help="0 means run in current process.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--input-unit-scale-to-m", type=float, default=0.001)
    p.add_argument("--enable-auto-scale", action="store_true", default=True)
    p.add_argument("--disable-auto-scale", action="store_true")
    p.add_argument("--scale-low", type=float, default=2.0)
    p.add_argument("--scale-high", type=float, default=6.0)
    p.add_argument("--scale-seed", type=int, default=42)
    p.add_argument("--manual-scale-factor", type=float, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    auto_scale = False if args.disable_auto_scale else args.enable_auto_scale

    cfg = DatasetGenerationConfig(
        layout_dir=args.layout_dir,
        output_dir=args.output_dir,
        samples_per_layout=args.samples_per_layout,
        samples_per_task=args.samples_per_task,
        progress_log_interval=args.progress_log_interval,
        sampling_method=args.sampling_method,
        storage_format=args.storage_format,
        builder_name=args.builder,
        num_modes=args.num_modes,
        max_workers=args.max_workers,
        seed=args.seed,
        overwrite=args.overwrite,
        input_unit_scale_to_m=args.input_unit_scale_to_m,
        enable_auto_scale=auto_scale,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        scale_seed=args.scale_seed,
        manual_scale_factor=args.manual_scale_factor,
    )

    summaries = generate_structural_dataset(cfg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "layouts": [asdict(item) for item in summaries],
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    total_layouts = len(summaries)
    total_samples = sum(item.total_samples for item in summaries)
    total_converged = sum(item.converged_samples for item in summaries)
    total_feasible = sum(item.feasible_samples for item in summaries)
    print(f"Layouts: {total_layouts}")
    print(f"Samples: {total_samples}")
    print(f"Converged: {total_converged}")
    print(f"Feasible: {total_feasible}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
