from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GroupSplitConfig:
    sample_dataset_path: str = (
        r"data\parametric\surrogate_dataset\surrogate_samples_with_layout_features.parquet"
    )
    output_dir: str = r"data\parametric\surrogate_dataset\splits"
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    n_splits: int = 5


def build_group_splits(cfg: GroupSplitConfig) -> dict[str, int | str | float]:
    if cfg.train_ratio <= 0.0 or cfg.val_ratio <= 0.0 or cfg.train_ratio + cfg.val_ratio >= 1.0:
        raise ValueError("Expected 0 < train_ratio, 0 < val_ratio, and train_ratio + val_ratio < 1.")
    if cfg.n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    df = pd.read_parquet(cfg.sample_dataset_path)
    if "layout_id" not in df.columns or "sample_id" not in df.columns:
        raise ValueError("Dataset must contain layout_id and sample_id columns.")

    layout_ids = sorted(str(x) for x in df["layout_id"].dropna().unique())
    if len(layout_ids) < 3:
        raise ValueError("Need at least 3 layouts for train/val/test split.")

    rng = random.Random(cfg.seed)
    shuffled = layout_ids.copy()
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = max(1, int(round(n_total * cfg.train_ratio)))
    n_val = max(1, int(round(n_total * cfg.val_ratio)))
    n_test = n_total - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_train >= n_val:
            n_train -= 1
        else:
            n_val -= 1

    train_layouts = sorted(shuffled[:n_train])
    val_layouts = sorted(shuffled[n_train : n_train + n_val])
    test_layouts = sorted(shuffled[n_train + n_val :])

    split_map: dict[str, str] = {}
    for lid in train_layouts:
        split_map[lid] = "train"
    for lid in val_layouts:
        split_map[lid] = "val"
    for lid in test_layouts:
        split_map[lid] = "test"

    fold_assignments: list[dict[str, int | str]] = []
    fold_layouts: dict[str, list[str]] = {f"fold_{i}": [] for i in range(cfg.n_splits)}
    for i, lid in enumerate(shuffled):
        fold = i % cfg.n_splits
        fold_assignments.append({"layout_id": lid, "group_fold": fold})
        fold_layouts[f"fold_{fold}"].append(lid)

    sample_split = df[["layout_id", "sample_id"]].copy()
    sample_split["split"] = sample_split["layout_id"].map(split_map)

    fold_df = pd.DataFrame(fold_assignments)
    sample_split = sample_split.merge(fold_df, on="layout_id", how="left", validate="many_to_one")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_json = {
        "config": asdict(cfg),
        "num_layouts": n_total,
        "num_samples": int(len(df)),
        "layout_split": {
            "train": train_layouts,
            "val": val_layouts,
            "test": test_layouts,
        },
        "fold_layouts": fold_layouts,
    }

    layout_split_path = out_dir / "layout_split.json"
    layout_split_path.write_text(json.dumps(split_json, ensure_ascii=True, indent=2), encoding="utf-8")

    layout_fold_path = out_dir / "layout_group_folds.parquet"
    fold_df.sort_values("layout_id").to_parquet(layout_fold_path, index=False)

    sample_split_path = out_dir / "sample_splits.parquet"
    sample_split.sort_values(["layout_id", "sample_id"]).to_parquet(sample_split_path, index=False)

    merged_path = out_dir / "surrogate_samples_with_splits.parquet"
    merged_df = df.merge(sample_split, on=["layout_id", "sample_id"], how="left", validate="one_to_one")
    merged_df.to_parquet(merged_path, index=False)

    return {
        "dataset": cfg.sample_dataset_path,
        "layout_split_json": str(layout_split_path),
        "layout_fold_table": str(layout_fold_path),
        "sample_split_table": str(sample_split_path),
        "merged_dataset": str(merged_path),
        "num_layouts": n_total,
        "num_samples": int(len(df)),
        "train_layouts": len(train_layouts),
        "val_layouts": len(val_layouts),
        "test_layouts": len(test_layouts),
        "n_splits": cfg.n_splits,
    }
