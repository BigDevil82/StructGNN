from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.surrogate.gnn.dataset import CAT_COLS, FEATURE_COLS, NUM_COLS, ParamPreprocessor
from src.surrogate.gnn.model import ParamEncoder, auto_param_emb_dims

TARGET = "material_steel_kg"
GROUP_COLS = ["layout_id", "N", "hs", "h_story", "intensity", "site_class", "seismic_group"]


@dataclass(frozen=True)
class PairwiseRankConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    output_dir: str = r"data\parametric\ckpt\steel_pairwise_ranker"
    seed: int = 42
    batch_size: int = 2048
    max_epochs: int = 2
    pairs_per_epoch: int = 300000
    eval_pairs: int = 200000
    min_pair_gap_kg: float = 3000.0
    large_gap_kg: float = 10000.0
    lr: float = 5.0e-4
    weight_decay: float = 1.0e-4
    hidden_dim: int = 128
    dropout: float = 0.1
    top_frac: float = 0.1
    log_interval: int = 50


class ScoreMLP(nn.Module):
    def __init__(self, graph_feat_dim: int, cat_cardinalities: list[int], hidden_dim: int, dropout: float):
        super().__init__()
        self.param_encoder = ParamEncoder(
            num_dim=len(NUM_COLS),
            cat_cardinalities=cat_cardinalities,
            emb_dims=auto_param_emb_dims(cat_cardinalities),
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + graph_feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, graph_feat: torch.Tensor) -> torch.Tensor:
        p = self.param_encoder(x_num, x_cat)
        return self.head(torch.cat([p, graph_feat], dim=1)).squeeze(1)


class SplitData:
    def __init__(
        self,
        df: pd.DataFrame,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        graph_feat: np.ndarray,
    ):
        self.df = df.reset_index(drop=True)
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.graph_feat = torch.tensor(graph_feat, dtype=torch.float32)
        self.y = self.df[TARGET].to_numpy(dtype=np.float64)
        self.groups = _make_groups(self.df)


def main() -> None:
    args = build_parser().parse_args()
    cfg = PairwiseRankConfig(**vars(args))
    metrics = run(cfg)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


def run(cfg: PairwiseRankConfig) -> dict[str, object]:
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    train, val, test, pre, graph_feat_dim = load_splits(cfg)
    model = ScoreMLP(
        graph_feat_dim=graph_feat_dim,
        cat_cardinalities=[len(pre.cat_vocab[c]) for c in CAT_COLS],
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        loss = train_epoch(model, train, cfg, opt, device, rng)
        val_metrics = eval_pairwise(model, val, cfg, device, rng)
        history.append({"epoch": epoch, "train_loss": loss, "val": val_metrics})
        print(
            f"[steel-rank] epoch={epoch}/{cfg.max_epochs} loss={loss:.6f} "
            f"val_pair_acc={val_metrics['pair_acc']:.4f} "
            f"val_large_gap_acc={val_metrics['large_gap_acc']:.4f} "
            f"val_top_recall={val_metrics['top_recall']:.4f} "
            f"val_regret={val_metrics['regret_mean_kg']:.1f}"
        )

    val_metrics = eval_pairwise(model, val, cfg, device, rng)
    test_metrics = eval_pairwise(model, test, cfg, device, rng)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_predictions(model, test, device, out_dir / "scores_test.parquet")
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "preprocess": pre.export(),
            "graph_feat_dim": int(graph_feat_dim),
            "hidden_dim": int(cfg.hidden_dim),
            "dropout": float(cfg.dropout),
            "target": TARGET,
            "group_cols": GROUP_COLS,
            "score_convention": "lower score means lower predicted steel usage",
        },
        out_dir / "steel_pairwise_ranker.pt",
    )

    metrics = {
        "config": asdict(cfg),
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    return metrics


def load_splits(cfg: PairwiseRankConfig) -> tuple[SplitData, SplitData, SplitData, ParamPreprocessor, int]:
    cols = ["layout_id", "split", "sample_id", TARGET, "hs", "h_story"] + FEATURE_COLS
    df = pd.read_parquet(cfg.dataset_path, columns=cols)
    graph_map = load_graph_features(cfg.graph_cache_dir)
    missing = sorted(set(df["layout_id"].astype(str)) - set(graph_map))
    if missing:
        raise ValueError(f"Missing graph cache for layouts: {missing[:5]}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    pre = ParamPreprocessor(NUM_COLS, CAT_COLS)
    pre.fit(train_df)
    x_train_num, x_train_cat = pre.transform(train_df)
    x_val_num, x_val_cat = pre.transform(val_df)
    x_test_num, x_test_cat = pre.transform(test_df)

    g_train = _graph_array(train_df, graph_map)
    g_val = _graph_array(val_df, graph_map)
    g_test = _graph_array(test_df, graph_map)
    return (
        SplitData(train_df, x_train_num, x_train_cat, g_train),
        SplitData(val_df, x_val_num, x_val_cat, g_val),
        SplitData(test_df, x_test_num, x_test_cat, g_test),
        pre,
        int(g_train.shape[1]),
    )


def train_epoch(model, data: SplitData, cfg: PairwiseRankConfig, opt, device: torch.device, rng) -> float:
    model.train()
    losses = []
    steps = max(1, cfg.pairs_per_epoch // cfg.batch_size)
    for step in range(steps):
        i, j, label, weight = sample_pairs(data, cfg.batch_size, cfg.min_pair_gap_kg, rng, weighted=True)
        i_t = torch.tensor(i, dtype=torch.long)
        j_t = torch.tensor(j, dtype=torch.long)
        idx = torch.cat([i_t, j_t], dim=0)
        idx = idx.to(device)
        label_t = torch.tensor(label, dtype=torch.float32, device=device)
        weight_t = torch.tensor(weight, dtype=torch.float32, device=device)

        opt.zero_grad(set_to_none=True)
        scores = model(
            data.x_num[idx.cpu()].to(device),
            data.x_cat[idx.cpu()].to(device),
            data.graph_feat[idx.cpu()].to(device),
        )
        s_i, s_j = scores.chunk(2, dim=0)
        logits = s_j - s_i
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label_t, weight=weight_t)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))

        if cfg.log_interval > 0 and (step + 1) % cfg.log_interval == 0:
            print(f"\rpair batch {step + 1}/{steps}, loss={losses[-1]:.4f}", end="", flush=True)
    if cfg.log_interval > 0:
        print()
    return float(np.mean(losses))


def eval_pairwise(model, data: SplitData, cfg: PairwiseRankConfig, device: torch.device, rng) -> dict[str, float]:
    scores = predict_scores(model, data, device)
    i, j, _, _ = sample_pairs(data, cfg.eval_pairs, 0.0, rng, weighted=False)
    y_i = data.y[i]
    y_j = data.y[j]
    s_i = scores[i]
    s_j = scores[j]
    gap = np.abs(y_i - y_j)
    true_i_better = y_i < y_j
    pred_i_better = s_i < s_j
    large = gap >= cfg.large_gap_kg
    metrics = {
        "pair_acc": float(np.mean(pred_i_better == true_i_better)),
        "large_gap_acc": float(np.mean((pred_i_better == true_i_better)[large])) if large.any() else float("nan"),
        "large_gap_rate": float(np.mean(large)),
        "spearman_group_mean": float(np.nanmean(group_spearman(data, scores))),
    }
    metrics.update(topk_metrics(data, scores, cfg.top_frac))
    return metrics


def sample_pairs(data: SplitData, n_pairs: int, min_gap: float, rng, *, weighted: bool) -> tuple[np.ndarray, ...]:
    group_ids = list(data.groups.keys())
    out_i = []
    out_j = []
    labels = []
    weights = []
    attempts = 0
    while len(out_i) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        group = data.groups[group_ids[int(rng.integers(0, len(group_ids)))]]
        a, b = rng.choice(group, size=2, replace=False)
        diff = data.y[a] - data.y[b]
        if abs(diff) < min_gap:
            continue
        out_i.append(a)
        out_j.append(b)
        labels.append(float(diff < 0.0))
        if weighted:
            weights.append(float(np.clip(abs(diff) / 20000.0, 0.5, 3.0)))
        else:
            weights.append(1.0)
    if len(out_i) < n_pairs:
        raise RuntimeError(f"Only sampled {len(out_i)} pairs; reduce min_pair_gap_kg")
    return (
        np.array(out_i, dtype=np.int64),
        np.array(out_j, dtype=np.int64),
        np.array(labels, dtype=np.float32),
        np.array(weights, dtype=np.float32),
    )


def predict_scores(model, data: SplitData, device: torch.device, batch_size: int = 8192) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(data.df), batch_size):
            idx = slice(start, start + batch_size)
            pred = model(
                data.x_num[idx].to(device),
                data.x_cat[idx].to(device),
                data.graph_feat[idx].to(device),
            )
            scores.append(pred.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)


def score_predictions(model, data: SplitData, device: torch.device, output_path: Path) -> None:
    scores = predict_scores(model, data, device)
    out = data.df[["layout_id", "sample_id", TARGET]].copy()
    out["rank_score"] = scores
    out.to_parquet(output_path, index=False)


def topk_metrics(data: SplitData, scores: np.ndarray, top_frac: float) -> dict[str, float]:
    recalls = []
    regrets = []
    for group in data.groups.values():
        if len(group) < 3:
            continue
        k = max(1, int(np.ceil(len(group) * top_frac)))
        true_order = group[np.argsort(data.y[group])]
        pred_order = group[np.argsort(scores[group])]
        true_top = set(true_order[:k].tolist())
        pred_top = set(pred_order[:k].tolist())
        recalls.append(len(true_top & pred_top) / k)
        regrets.append(float(data.y[pred_order[:k]].min() - data.y[true_order[0]]))
    return {
        "top_recall": float(np.mean(recalls)),
        "regret_mean_kg": float(np.mean(regrets)),
        "regret_p90_kg": float(np.quantile(regrets, 0.9)),
    }


def group_spearman(data: SplitData, scores: np.ndarray) -> list[float]:
    vals = []
    for group in data.groups.values():
        if len(group) < 3:
            continue
        y_rank = pd.Series(data.y[group]).rank(method="average").to_numpy()
        s_rank = pd.Series(scores[group]).rank(method="average").to_numpy()
        vals.append(float(np.corrcoef(y_rank, s_rank)[0, 1]))
    return vals


def _make_groups(df: pd.DataFrame) -> dict[tuple, np.ndarray]:
    groups = {}
    for key, idx in df.groupby(GROUP_COLS, observed=True).indices.items():
        arr = np.array(idx, dtype=np.int64)
        if len(arr) >= 2:
            groups[key] = arr
    return groups


def load_graph_features(graph_cache_dir: str) -> dict[str, np.ndarray]:
    out = {}
    for path in Path(graph_cache_dir).glob("*.pt"):
        g = torch.load(path, map_location="cpu", weights_only=True)
        out[path.stem] = g["graph_feat"].reshape(-1).numpy().astype(np.float32)
    return out


def _graph_array(df: pd.DataFrame, graph_map: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([graph_map[str(layout_id)] for layout_id in df["layout_id"]], axis=0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train pairwise ranker for steel usage comparison.")
    p.add_argument("--dataset-path", default=PairwiseRankConfig.dataset_path)
    p.add_argument("--graph-cache-dir", default=PairwiseRankConfig.graph_cache_dir)
    p.add_argument("--output-dir", default=PairwiseRankConfig.output_dir)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--max-epochs", type=int, default=2)
    p.add_argument("--pairs-per-epoch", type=int, default=300000)
    p.add_argument("--eval-pairs", type=int, default=200000)
    p.add_argument("--min-pair-gap-kg", type=float, default=3000.0)
    p.add_argument("--large-gap-kg", type=float, default=10000.0)
    p.add_argument("--lr", type=float, default=5.0e-4)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--log-interval", type=int, default=50)
    return p


if __name__ == "__main__":
    main()
