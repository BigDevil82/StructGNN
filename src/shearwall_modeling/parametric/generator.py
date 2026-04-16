import json
import logging
import math
import random
import threading
import zlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import openseespy.opensees as ops
import pandas as pd

from ...misc.parallel import run_batch
from ..analysis.evaluation import AnalysisResultBuilder, SeismicEvaluationPipeline
from ..builders import DetailedShellBuilder, EquivalentFrameBuilder, MVLEMFrameBuilder
from ..builders.base import AnalysisModelContext, StructuralModelBuilder
from ..core.config import MaterialConfig, ModelConfig, SectionConfig, SeismicConfig, StandardStoryGroupConfig
from ..core.domain import FEMInput
from ..design import ReinforcementDesignPipeline
from ..geometry.scaling import load_and_scale_input

SamplingMethod = Literal["random", "lhs"]
StorageFormat = Literal["parquet", "hdf5"]
BuilderName = Literal["detailed_shell", "equivalent_frame", "mvlem_frame"]


@dataclass(frozen=True)
class ParametricModelParams:
    N: int
    tw_bot: int
    tw_mid: int
    tw_top: int
    hb_main: int
    bb_main: int
    hb_sec: int
    bb_sec: int
    hs: int
    conc_bot: str
    conc_mid: str
    conc_top: str
    intensity: float
    site_class: str
    seismic_group: int
    h_story: float = 2.9


PARAM_SPACE: dict[str, list[Any]] = {
    "N": list(range(18, 34)),
    "tw_bot": [200, 250, 300, 350, 400],
    "tw_mid": [160, 180, 200, 250, 300],
    "tw_top": [160, 180, 200, 250],
    "hb_main": [400, 500, 550, 600, 650, 700],
    "bb_main": [200, 250, 300, 350],
    "hb_sec": [300, 400, 450, 500],
    "bb_sec": [200, 250, 300],
    "hs": [100, 120, 150],
    "conc_bot": ["C30", "C35", "C40", "C45", "C50"],
    "conc_mid": ["C30", "C35", "C40", "C45"],
    "conc_top": ["C30", "C35", "C40"],
    "intensity": [6.0, 7.0, 7.5, 8.0],
    "site_class": ["I0", "I", "II", "III", "IV"],
    "seismic_group": [1, 2, 3],
    "h_story": [2.8, 2.9, 3.0],
}


@dataclass(frozen=True)
class DatasetGenerationConfig:
    layout_dir: str = r"data\dxf\cad_json_data\fem_raw"
    output_dir: str = r"outputs\result\parametric_dataset"
    samples_per_layout: int = 500
    samples_per_task: int = 20
    progress_log_interval: int = 200
    sampling_method: SamplingMethod = "lhs"
    storage_format: StorageFormat = "parquet"
    builder_name: BuilderName = "mvlem_frame"
    num_modes: int = 6
    max_workers: int | None = None
    seed: int = 42
    overwrite: bool = False
    input_unit_scale_to_m: float = 0.001
    enable_auto_scale: bool = True
    scale_low: float = 2.0
    scale_high: float = 6.0
    scale_seed: int = 42
    manual_scale_factor: float | None = None


@dataclass(frozen=True)
class LayoutGenerationTask:
    layout_path: str
    output_dir: str
    cfg: DatasetGenerationConfig


@dataclass(frozen=True)
class SampleChunkTask:
    layout_path: str
    layout_id: str
    cfg: DatasetGenerationConfig
    sample_start_idx: int
    params_chunk: list[ParametricModelParams]


@dataclass(frozen=True)
class SampleChunkResult:
    layout_id: str
    rows: list[dict[str, Any]]


@dataclass(frozen=True)
class LayoutDatasetSummary:
    layout_id: str
    output_path: str
    total_samples: int
    converged_samples: int
    feasible_samples: int
    skipped: bool = False
    error: str | None = None


def sample_parametric_model_params(rng: random.Random | None = None) -> ParametricModelParams:
    sampler = rng or random.Random()
    raw = {key: sampler.choice(values) for key, values in PARAM_SPACE.items()}
    return _normalize_params(raw)


def sample_parametric_model_params_batch(
    n: int,
    method: SamplingMethod = "lhs",
    seed: int = 42,
) -> list[ParametricModelParams]:
    if n <= 0:
        return []

    if method == "random":
        rng = random.Random(seed)
        return [sample_parametric_model_params(rng) for _ in range(n)]

    if method != "lhs":
        raise ValueError(f"Unsupported sampling_method={method}.")

    keys = list(PARAM_SPACE.keys())
    rng = random.Random(seed)
    idx_by_key: dict[str, list[int]] = {}
    for key in keys:
        levels = len(PARAM_SPACE[key])
        perm = list(range(n))
        rng.shuffle(perm)
        idxs: list[int] = []
        for i in range(n):
            u = (perm[i] + rng.random()) / n
            idx = int(math.floor(u * levels))
            idxs.append(min(levels - 1, max(0, idx)))
        idx_by_key[key] = idxs

    return [_normalize_params({key: PARAM_SPACE[key][idx_by_key[key][i]] for key in keys}) for i in range(n)]


def build_model_config_from_params(
    params: ParametricModelParams | dict[str, Any],
    *,
    story_height: float | None = None,
    num_modes: int = 6,
) -> ModelConfig:
    p = _normalize_params(params)
    h_story = story_height if story_height is not None else p.h_story
    n_bottom, n_middle, n_top = _split_stories(p.N)

    def make_section(tw_mm: int) -> SectionConfig:
        return SectionConfig(
            wall_thickness=tw_mm / 1000.0,
            beam_width=p.bb_main / 1000.0,
            beam_depth=p.hb_main / 1000.0,
            secondary_beam_width=p.bb_sec / 1000.0,
            secondary_beam_depth=p.hb_sec / 1000.0,
            slab_thickness=p.hs / 1000.0,
        )

    groups = [
        StandardStoryGroupConfig(
            count=n_bottom,
            story_height=h_story,
            section=make_section(p.tw_bot),
            material=MaterialConfig(concrete_grade=p.conc_bot),
        ),
        StandardStoryGroupConfig(
            count=n_middle,
            story_height=h_story,
            section=make_section(p.tw_mid),
            material=MaterialConfig(concrete_grade=p.conc_mid),
        ),
        StandardStoryGroupConfig(
            count=n_top,
            story_height=h_story,
            section=make_section(p.tw_top),
            material=MaterialConfig(concrete_grade=p.conc_top),
        ),
    ]

    return ModelConfig(
        standard_story_groups=groups,
        seismic=SeismicConfig(
            intensity=p.intensity,
            site_class=p.site_class,
            seismic_group=p.seismic_group,
        ),
        num_modes=num_modes,
    )


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

    if cfg.max_workers != 0:
        progress_every = max(1, cfg.progress_log_interval)
        done_chunks_by_layout: dict[str, int] = defaultdict(int)
        progress_lock = threading.Lock()

        def on_outcome(outcome: Any, done: int, total: int) -> None:
            with progress_lock:
                lid = outcome.item.layout_id
                done_chunks_by_layout[lid] += 1
                if done == 1 or done == total or done % progress_every == 0:
                    print(
                        f"[progress] chunks {done}/{total}, "
                        f"layouts_done={sum(1 for k, v in done_chunks_by_layout.items() if v >= chunk_total_by_layout[k])}/{len(chunk_total_by_layout)}"
                    )
                if done_chunks_by_layout[lid] == chunk_total_by_layout[lid]:
                    print(f"[progress] layout {lid} chunk stage done ({done_chunks_by_layout[lid]} chunks)")

        outcomes = run_batch(
            tasks,
            _generate_one_sample_chunk,
            max_workers=cfg.max_workers,
            backend="process",
            on_outcome=on_outcome,
        )
        for outcome in outcomes:
            layout_id = outcome.item.layout_id
            if outcome.ok and outcome.result is not None:
                rows_by_layout[layout_id].extend(outcome.result.rows)
            else:
                errors_by_layout[layout_id].append(outcome.error or "Unknown chunk error")
    else:
        for task in tasks:
            try:
                chunk_result = _generate_one_sample_chunk(task)
                rows_by_layout[task.layout_id].extend(chunk_result.rows)
            except Exception as exc:
                errors_by_layout[task.layout_id].append(str(exc))

    summaries: list[LayoutDatasetSummary] = list(skipped_summaries.values())
    for lp in layout_paths:
        layout_id = Path(lp).stem
        if layout_id in skipped_summaries:
            continue

        output_path = output_path_by_layout[layout_id]
        rows = rows_by_layout.get(layout_id, [])
        errors = errors_by_layout.get(layout_id, [])
        if not rows:
            summaries.append(
                LayoutDatasetSummary(
                    layout_id=layout_id,
                    output_path=output_path,
                    total_samples=0,
                    converged_samples=0,
                    feasible_samples=0,
                    error=("; ".join(errors[:3]) if errors else "No rows generated."),
                )
            )
            continue

        rows.sort(key=lambda row: int(row["sample_id"]))
        _write_rows(rows, Path(output_path), cfg.storage_format)
        converged_count = sum(1 for row in rows if row["converged"])
        feasible_count = sum(1 for row in rows if row["feasible"])
        summaries.append(
            LayoutDatasetSummary(
                layout_id=layout_id,
                output_path=output_path,
                total_samples=len(rows),
                converged_samples=converged_count,
                feasible_samples=feasible_count,
                error=("; ".join(errors[:3]) if errors else None),
            )
        )

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
    rows = [
        _analyze_one_sample(
            input_data=input_data,
            params=params,
            sample_id=task.sample_start_idx + i,
            layout_id=task.layout_id,
            geom_scale=scale,
            cfg=cfg,
        )
        for i, params in enumerate(task.params_chunk)
    ]
    return SampleChunkResult(layout_id=task.layout_id, rows=rows)


def _generate_one_layout_dataset(task: LayoutGenerationTask) -> LayoutDatasetSummary:
    cfg = task.cfg
    layout_path = Path(task.layout_path)
    layout_id = layout_path.stem
    ext = "parquet" if cfg.storage_format == "parquet" else "h5"
    output_path = Path(task.output_dir) / f"{layout_id}.{ext}"

    if output_path.exists() and not cfg.overwrite:
        return LayoutDatasetSummary(
            layout_id=layout_id,
            output_path=str(output_path),
            total_samples=0,
            converged_samples=0,
            feasible_samples=0,
            skipped=True,
        )

    layout_seed = cfg.seed + zlib.crc32(layout_id.encode("utf-8"))
    params_list = sample_parametric_model_params_batch(
        n=cfg.samples_per_layout,
        method=cfg.sampling_method,
        seed=layout_seed,
    )

    input_data, scale = load_and_scale_input(
        json_path=layout_path,
        input_unit_scale_to_m=cfg.input_unit_scale_to_m,
        enable_auto_scale=cfg.enable_auto_scale,
        low=cfg.scale_low,
        high=cfg.scale_high,
        seed=cfg.scale_seed,
        manual_factor=cfg.manual_scale_factor,
    )

    rows = [
        _analyze_one_sample(
            input_data=input_data,
            params=params,
            sample_id=i,
            layout_id=layout_id,
            geom_scale=scale,
            cfg=cfg,
        )
        for i, params in enumerate(params_list, start=1)
    ]

    _write_rows(rows, output_path, cfg.storage_format)
    converged_count = sum(1 for row in rows if row["converged"])
    feasible_count = sum(1 for row in rows if row["feasible"])
    return LayoutDatasetSummary(
        layout_id=layout_id,
        output_path=str(output_path),
        total_samples=len(rows),
        converged_samples=converged_count,
        feasible_samples=feasible_count,
    )


def _analyze_one_sample(
    input_data: FEMInput,
    params: ParametricModelParams,
    sample_id: int,
    layout_id: str,
    geom_scale: float,
    cfg: DatasetGenerationConfig,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "layout_id": layout_id,
        "sample_id": sample_id,
        "geom_scale": geom_scale,
        **asdict(params),
    }
    logger = logging.getLogger("shearwall_dataset")
    logger.setLevel(logging.WARN)
    builder = _create_builder(cfg.builder_name, logger)

    try:
        model_cfg = build_model_config_from_params(params, num_modes=cfg.num_modes)
        build_result = builder.build(input_data, model_cfg)
        context = AnalysisModelContext(build_result=build_result, config=model_cfg, logger=logger)
        analysis_result = AnalysisResultBuilder(context).build()
        overall_result, dir_results = SeismicEvaluationPipeline(model_cfg).evaluate(analysis_result)

        periods = analysis_result.modal_summary.periods[:6]
        while len(periods) < 6:
            periods.append(float("nan"))

        all_metrics = [m for result in dir_results.values() for m in result.metrics]
        wall_axial: dict[int, float] = {}
        wall_shear: dict[int, float] = {}
        beam_shear: dict[int, float] = {}
        for metric in analysis_result.wall_uls_metrics:
            wall_axial[metric.wall_id] = max(wall_axial.get(metric.wall_id, 0.0), metric.axial_ratio)
            wall_shear[metric.wall_id] = max(wall_shear.get(metric.wall_id, 0.0), metric.shear_pressure_ratio)
        for metric in analysis_result.beam_uls_metrics:
            beam_shear[metric.beam_id] = max(beam_shear.get(metric.beam_id, 0.0), metric.shear_pressure_ratio)

        mass_total_kg = sum(build_result.floor_load_masses) + sum(build_result.floor_self_masses)
        design_summary = ReinforcementDesignPipeline(
            build_result=build_result,
            analysis_result=analysis_result,
            config=model_cfg,
            logger=logger,
        ).run()
        row.update(
            {
                "converged": True,
                "feasible": overall_result.is_passed,
                "design_passed": design_summary.is_passed,
                "design_failed_beam_count": len(design_summary.failed_beams),
                "design_failed_wall_count": len(design_summary.failed_walls),
                "material_concrete_kg": design_summary.total_concrete_kg,
                "material_steel_kg": design_summary.total_steel_kg,
                "mass_total_t": mass_total_kg / 1000.0,
                "T1": periods[0],
                "T2": periods[1],
                "T3": periods[2],
                "T4": periods[3],
                "T5": periods[4],
                "T6": periods[5],
                "period_ratio": analysis_result.modal_summary.period_ratio,
                "torsion_ratio": max(m.torsion_ratio for m in all_metrics),
                "min_shear_weight_ratio": min(m.shear_weight_ratio for m in all_metrics),
                "max_drift_ratio": max(result.max_interstory_drift_ratio for result in dir_results.values()),
                "min_stiffness_ratio": min(m.stiffness_ratio_adjacent for m in all_metrics),
                "wall_axial_ratios": json.dumps(wall_axial, ensure_ascii=True, separators=(",", ":")),
                "wall_shear_ratios": json.dumps(wall_shear, ensure_ascii=True, separators=(",", ":")),
                "beam_shear_ratios": json.dumps(beam_shear, ensure_ascii=True, separators=(",", ":")),
                "error": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "converged": False,
                "feasible": False,
                "design_passed": False,
                "design_failed_beam_count": -1,
                "design_failed_wall_count": -1,
                "material_concrete_kg": float("nan"),
                "material_steel_kg": float("nan"),
                "mass_total_t": float("nan"),
                "T1": float("nan"),
                "T2": float("nan"),
                "T3": float("nan"),
                "T4": float("nan"),
                "T5": float("nan"),
                "T6": float("nan"),
                "period_ratio": float("nan"),
                "torsion_ratio": float("nan"),
                "min_shear_weight_ratio": float("nan"),
                "max_drift_ratio": float("nan"),
                "min_stiffness_ratio": float("nan"),
                "wall_axial_ratios": "{}",
                "wall_shear_ratios": "{}",
                "beam_shear_ratios": "{}",
                "error": str(exc),
            }
        )
    finally:
        ops.wipe()

    return row


def _create_builder(name: BuilderName, logger: logging.Logger) -> StructuralModelBuilder:
    if name == "mvlem_frame":
        return MVLEMFrameBuilder(logger, fiber_width=0.2)
    if name == "equivalent_frame":
        return EquivalentFrameBuilder(logger)
    if name == "detailed_shell":
        return DetailedShellBuilder(logger)
    raise ValueError(f"Unsupported builder_name={name}.")


def _write_rows(rows: list[dict[str, Any]], output_path: Path, fmt: StorageFormat) -> None:
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


def _split_stories(total_stories: int) -> tuple[int, int, int]:
    if total_stories < 3:
        raise ValueError("N must be at least 3.")
    n1 = min(10, max(2, int(math.floor(0.3 * total_stories))))
    n2 = max(n1 + 1, int(math.floor(0.7 * total_stories)))
    n2 = min(total_stories - 1, n2)
    return n1, n2 - n1, total_stories - n2


def _normalize_conc(grade: str) -> str:
    g = grade.strip().upper()
    return g if g.startswith("C") else f"C{g}"


def _normalize_params(raw: ParametricModelParams | dict[str, Any]) -> ParametricModelParams:
    if isinstance(raw, ParametricModelParams):
        p = raw
    else:
        p = ParametricModelParams(
            N=int(raw["N"]),
            tw_bot=int(raw["tw_bot"]),
            tw_mid=int(raw["tw_mid"]),
            tw_top=int(raw["tw_top"]),
            hb_main=int(raw["hb_main"]),
            bb_main=int(raw["bb_main"]),
            hb_sec=int(raw["hb_sec"]),
            bb_sec=int(raw["bb_sec"]),
            hs=int(raw["hs"]),
            conc_bot=_normalize_conc(str(raw["conc_bot"])),
            conc_mid=_normalize_conc(str(raw["conc_mid"])),
            conc_top=_normalize_conc(str(raw["conc_top"])),
            intensity=float(raw["intensity"]),
            site_class=str(raw["site_class"]).upper(),
            seismic_group=int(raw["seismic_group"]),
            h_story=float(raw.get("h_story", 2.9)),
        )

    grades = ["C30", "C35", "C40", "C45", "C50"]
    i_bot = grades.index(p.conc_bot)
    i_mid = min(grades.index(p.conc_mid), i_bot)
    i_top = min(grades.index(p.conc_top), i_mid)
    tw_mid = min(p.tw_mid, p.tw_bot)
    tw_top = min(p.tw_top, tw_mid)

    return ParametricModelParams(
        N=p.N,
        tw_bot=p.tw_bot,
        tw_mid=tw_mid,
        tw_top=tw_top,
        hb_main=p.hb_main,
        bb_main=p.bb_main,
        hb_sec=p.hb_sec,
        bb_sec=p.bb_sec,
        hs=p.hs,
        conc_bot=grades[i_bot],
        conc_mid=grades[i_mid],
        conc_top=grades[i_top],
        intensity=p.intensity,
        site_class=p.site_class,
        seismic_group=p.seismic_group,
        h_story=p.h_story,
    )
