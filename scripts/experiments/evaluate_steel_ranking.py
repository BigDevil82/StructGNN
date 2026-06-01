from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GROUP_COLS = ["layout_id", "N", "hs", "h_story", "intensity", "site_class", "seismic_group"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate steel prediction files as rankers.")
    p.add_argument("--dataset-path", default=r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet")
    p.add_argument("--predictions-path", required=True)
    p.add_argument("--score-col", required=True)
    p.add_argument("--output-path", default=None)
    p.add_argument("--eval-pairs", type=int, default=200000)
    p.add_argument("--large-gap-kg", type=float, default=10000.0)
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=123)
    return p


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate(
        dataset_path=args.dataset_path,
        predictions_path=args.predictions_path,
        score_col=args.score_col,
        eval_pairs=args.eval_pairs,
        large_gap_kg=args.large_gap_kg,
        top_frac=args.top_frac,
        seed=args.seed,
    )
    text = json.dumps(metrics, ensure_ascii=True, indent=2)
    print(text)
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text, encoding="utf-8")


def evaluate(
    *,
    dataset_path: str,
    predictions_path: str,
    score_col: str,
    eval_pairs: int,
    large_gap_kg: float,
    top_frac: float,
    seed: int,
) -> dict[str, float]:
    cols = ["layout_id", "sample_id", "split", *GROUP_COLS[1:]]
    meta = pd.read_parquet(dataset_path, columns=cols)
    meta = meta[meta["split"] == "test"].copy()
    pred = pd.read_parquet(predictions_path)
    if "material_steel_kg" in pred.columns and "steel_true_kg" not in pred.columns:
        pred = pred.rename(columns={"material_steel_kg": "steel_true_kg"})
    df = pred.merge(meta, on=["layout_id", "sample_id"], how="left", validate="one_to_one")
    if df[score_col].isna().any():
        raise ValueError(f"Missing score_col values: {score_col}")
    groups = _groups(df)
    y = df["steel_true_kg"].to_numpy(dtype=float)
    score = df[score_col].to_numpy(dtype=float)
    i, j = _sample_pairs(groups, eval_pairs, seed)
    gap = np.abs(y[i] - y[j])
    ok = (score[i] < score[j]) == (y[i] < y[j])
    large = gap >= large_gap_kg
    out = {
        "pair_acc": float(ok.mean()),
        "large_gap_acc": float(ok[large].mean()) if large.any() else float("nan"),
        "large_gap_rate": float(large.mean()),
        "spearman_group_mean": float(np.nanmean(_group_spearman(groups, y, score))),
    }
    out.update(_topk(groups, y, score, top_frac))
    return out


def _groups(df: pd.DataFrame) -> dict[tuple, np.ndarray]:
    return {
        k: np.array(v, dtype=np.int64)
        for k, v in df.groupby(GROUP_COLS, observed=True).indices.items()
        if len(v) >= 2
    }


def _sample_pairs(groups: dict[tuple, np.ndarray], n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    keys = list(groups.keys())
    i = []
    j = []
    for _ in range(n):
        group = groups[keys[int(rng.integers(0, len(keys)))]]
        a, b = rng.choice(group, size=2, replace=False)
        i.append(a)
        j.append(b)
    return np.array(i, dtype=np.int64), np.array(j, dtype=np.int64)


def _topk(groups: dict[tuple, np.ndarray], y: np.ndarray, score: np.ndarray, top_frac: float) -> dict[str, float]:
    recalls = []
    regrets = []
    for group in groups.values():
        if len(group) < 3:
            continue
        k = max(1, int(np.ceil(len(group) * top_frac)))
        true_order = group[np.argsort(y[group])]
        pred_order = group[np.argsort(score[group])]
        recalls.append(len(set(true_order[:k].tolist()) & set(pred_order[:k].tolist())) / k)
        regrets.append(float(y[pred_order[:k]].min() - y[true_order[0]]))
    return {
        "top_recall": float(np.mean(recalls)),
        "regret_mean_kg": float(np.mean(regrets)),
        "regret_p90_kg": float(np.quantile(regrets, 0.9)),
    }


def _group_spearman(groups: dict[tuple, np.ndarray], y: np.ndarray, score: np.ndarray) -> list[float]:
    vals = []
    for group in groups.values():
        if len(group) < 3:
            continue
        y_rank = pd.Series(y[group]).rank(method="average").to_numpy()
        s_rank = pd.Series(score[group]).rank(method="average").to_numpy()
        vals.append(float(np.corrcoef(y_rank, s_rank)[0, 1]))
    return vals


if __name__ == "__main__":
    main()
