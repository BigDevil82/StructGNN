import json
import threading
import zlib
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ...misc.parallel import run_batch
from ..geometry.scaling import load_and_scale_input
from .analysis import analyze_parametric_model
from .space import PARAM_SPACE, sample_parametric_model_params_batch
from .types import DatasetGenerationConfig, LayoutDatasetSummary, SampleChunkResult, SampleChunkTask


def generate_structural_dataset(cfg: DatasetGenerationConfig) -> list[LayoutDatasetSummary]:
    layout_paths = sorted(str(p) for p in Path(cfg.layout_dir).glob("*.json"))
    if not layout_paths:
        raise ValueError(f"No layout json files found in {cfg.layout_dir}.")
    if cfg.samples_per_task <= 0:
        raise ValueError("samples_per_task must be >= 1.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = "parquet" if cfg.storage_format == "parquet" else "h5"
    tasks: list[SampleChunkTask] = []
    rows_by_layout: dict[str, list[dict[str, Any]]] = {}
    errors_by_layout: dict[str, list[str]] = {}
    chunk_total_by_layout: dict[str, int] = defaultdict(int)
    skipped_summaries: dict[str, LayoutDatasetSummary] = {}
    output_path_by_layout: dict[str, str] = {}

    for lp in layout_paths:
        layout_path = Path(lp)
        layout_id = layout_path.stem
        output_path = output_dir / f"{layout_id}.{ext}"
        output_path_by_layout[layout_id] = str(output_path)

        if output_path.exists() and not cfg.overwrite:
            skipped_summaries[layout_id] = LayoutDatasetSummary(
                layout_id=layout_id,
                output_path=str(output_path),
                total_samples=0,
                converged_samples=0,
                feasible_samples=0,
                skipped=True,
            )
            continue

        layout_seed = cfg.seed + zlib.crc32(layout_id.encode("utf-8"))
        params_list = sample_parametric_model_params_batch(
            n=cfg.samples_per_layout,
            method=cfg.sampling_method,
            seed=layout_seed,
        )
        rows_by_layout[layout_id] = []
        errors_by_layout[layout_id] = []

        for start in range(0, len(params_list), cfg.samples_per_task):
            chunk = params_list[start : start + cfg.samples_per_task]
            chunk_total_by_layout[layout_id] += 1
            tasks.append(
                SampleChunkTask(
                    layout_path=str(layout_path),
                    layout_id=layout_id,
                    cfg=cfg,
                    sample_start_idx=start + 1,
                    params_chunk=chunk,
                )
            )

    done_chunks_by_layout: dict[str, int] = defaultdict(int)
    summary_by_layout: dict[str, LayoutDatasetSummary] = {}
    progress_every = max(1, cfg.progress_log_interval)
    progress_lock = threading.Lock()

    def finalize_layout_if_ready(layout_id: str) -> None:
        if layout_id in summary_by_layout:
            return
        if done_chunks_by_layout[layout_id] < chunk_total_by_layout[layout_id]:
            return

        output_path = output_path_by_layout[layout_id]
        rows = rows_by_layout.get(layout_id, [])
        errors = errors_by_layout.get(layout_id, [])
        if not rows:
            summary_by_layout[layout_id] = LayoutDatasetSummary(
                layout_id=layout_id,
                output_path=output_path,
                total_samples=0,
                converged_samples=0,
                feasible_samples=0,
                error=("; ".join(errors[:3]) if errors else "No rows generated."),
            )
            print(f"[progress] layout {layout_id} saved with 0 rows")
            return

        rows.sort(key=lambda row: int(row["sample_id"]))
        _write_rows(rows, Path(output_path), cfg.storage_format)
        converged_count = sum(1 for row in rows if row["converged"])
        feasible_count = sum(1 for row in rows if row["feasible"])
        summary_by_layout[layout_id] = LayoutDatasetSummary(
            layout_id=layout_id,
            output_path=output_path,
            total_samples=len(rows),
            converged_samples=converged_count,
            feasible_samples=feasible_count,
            error=("; ".join(errors[:3]) if errors else None),
        )
        print(f"[progress] layout {layout_id} saved ({len(rows)} rows)")
        rows_by_layout.pop(layout_id, None)
        errors_by_layout.pop(layout_id, None)

    if cfg.max_workers != 0:

        def on_outcome(outcome: Any, done: int, total: int) -> None:
            with progress_lock:
                lid = outcome.item.layout_id
                done_chunks_by_layout[lid] += 1
                if outcome.ok and outcome.result is not None:
                    rows_by_layout[lid].extend(outcome.result.rows)
                else:
                    errors_by_layout[lid].append(outcome.error or "Unknown chunk error")

                if done == 1 or done == total or done % progress_every == 0:
                    print(
                        f"[progress] chunks {done}/{total}, "
                        f"layouts_saved={len(summary_by_layout)}/{len(chunk_total_by_layout)}"
                    )

                finalize_layout_if_ready(lid)

        run_batch(
            tasks,
            _generate_one_sample_chunk,
            max_workers=cfg.max_workers,
            backend="process",
            on_outcome=on_outcome,
        )
    else:
        total = len(tasks)
        for i, task in enumerate(tasks, start=1):
            lid = task.layout_id
            done_chunks_by_layout[lid] += 1
            try:
                chunk_result = _generate_one_sample_chunk(task)
                rows_by_layout[lid].extend(chunk_result.rows)
            except Exception as exc:
                errors_by_layout[lid].append(str(exc))

            if i == 1 or i == total or i % progress_every == 0:
                print(
                    f"[progress] chunks {i}/{total}, layouts_saved={len(summary_by_layout)}/{len(chunk_total_by_layout)}"
                )
            finalize_layout_if_ready(lid)

    summaries: list[LayoutDatasetSummary] = list(skipped_summaries.values())
    for lp in layout_paths:
        layout_id = Path(lp).stem
        if layout_id in skipped_summaries:
            continue
        finalize_layout_if_ready(layout_id)
        if layout_id in summary_by_layout:
            summaries.append(summary_by_layout[layout_id])

    _write_metadata(cfg, summaries)
    return summaries


def _generate_one_sample_chunk(task: SampleChunkTask) -> SampleChunkResult:
    cfg = task.cfg
    layout_path = Path(task.layout_path)
    input_data, scale = load_and_scale_input(
        json_path=layout_path,
        input_unit_scale_to_m=cfg.input_unit_scale_to_m,
        enable_auto_scale=cfg.enable_auto_scale,
        low=cfg.scale_low,
        high=cfg.scale_high,
        seed=cfg.scale_seed,
        manual_factor=cfg.manual_scale_factor,
    )

    rows = []
    for i, params in enumerate(task.params_chunk):
        result = analyze_parametric_model(input_data, params, cfg)
        rows.append(
            {
                "layout_id": task.layout_id,
                "sample_id": task.sample_start_idx + i,
                "geom_scale": scale,
                **asdict(params),
                "converged": result.converged,
                "feasible": result.feasible,
                "design_passed": result.design_passed,
                "design_failed_beam_count": result.design_failed_beam_count,
                "design_failed_wall_count": result.design_failed_wall_count,
                "material_concrete_kg": result.material_concrete_kg,
                "material_steel_kg": result.material_steel_kg,
                "mass_total_t": result.mass_total_t,
                "T1": result.T1,
                "T2": result.T2,
                "T3": result.T3,
                "T4": result.T4,
                "T5": result.T5,
                "T6": result.T6,
                "period_ratio": result.period_ratio,
                "torsion_ratio": result.torsion_ratio,
                "min_shear_weight_ratio": result.min_shear_weight_ratio,
                "max_drift_ratio": result.max_drift_ratio,
                "min_stiffness_ratio": result.min_stiffness_ratio,
                "wall_axial_ratios": json.dumps(
                    result.wall_axial_ratios, ensure_ascii=True, separators=(",", ":")
                ),
                "wall_shear_ratios": json.dumps(
                    result.wall_shear_ratios, ensure_ascii=True, separators=(",", ":")
                ),
                "beam_shear_ratios": json.dumps(
                    result.beam_shear_ratios, ensure_ascii=True, separators=(",", ":")
                ),
                "error": result.error,
            }
        )

    return SampleChunkResult(layout_id=task.layout_id, rows=rows)


def _write_rows(rows: list[dict[str, Any]], output_path: Path, fmt: str) -> None:
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(output_path, index=False)
        return
    if fmt == "hdf5":
        df.to_hdf(output_path, key="samples", mode="w", format="table")
        return
    raise ValueError(f"Unsupported storage_format={fmt}.")


def _write_metadata(cfg: DatasetGenerationConfig, summaries: list[LayoutDatasetSummary]) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "param_space": PARAM_SPACE,
        "layouts": [
            {
                "layout_id": s.layout_id,
                "output_path": s.output_path,
                "total_samples": s.total_samples,
                "converged_samples": s.converged_samples,
                "feasible_samples": s.feasible_samples,
                "skipped": s.skipped,
                "error": s.error,
            }
            for s in summaries
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
    )
