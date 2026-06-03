from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.surrogate.features.consts import LAYOUT_FEATURES

from .surrogate_screening import GNNFeasibilityScreener, SurrogateScreeningConfig


@dataclass(frozen=True)
class SteelRankingConfig:
    enabled: bool = False
    artifact_path: str = r"data\parametric\ckpt\steel_gnn_room_lr5e4_b512\gnn_steel.pt"
    graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    layout_features_path: str = r"data\parametric\surrogate_dataset\layout_features.parquet"
    eval_ratio: float = 0.4
    min_eval_count: int = 8
    random_ratio: float = 0.1
    batch_size: int = 512
    num_workers: int = 0
    skipped_objective: float = 1.0e12
    use_feasibility_penalty: bool = False
    feasibility_artifact_path: str = (
        r"data\parametric\ckpt\baseline_gnn_room_hybrid_h256_screen995_v1\gnn_final_pass.pt"
    )
    feasibility_graph_cache_dir: str = r"data\parametric\cache\gnn_room_graph_cache"
    feasibility_layout_features_path: str = r"data\parametric\surrogate_dataset\layout_features.parquet"
    feasibility_threshold: float | None = None
    feasibility_penalty_kg: float = 200000.0


@dataclass(frozen=True)
class SteelRankingPrediction:
    score: float
    steel_kg: float
    feasible_probability: float | None = None
    infeasible_risk: float = 0.0


class GNNSteelRanker:
    def __init__(
        self,
        cfg: SteelRankingConfig,
        *,
        layout_id: str,
        fixed_params: dict[str, Any],
    ):
        self.cfg = cfg
        self.layout_id = str(layout_id)
        self.fixed_params = dict(fixed_params)
        self._cache: dict[tuple[Any, ...], float] = {}
        self._rank_cache: dict[tuple[Any, ...], SteelRankingPrediction] = {}

        import torch
        from src.surrogate.gnn.dataset import ParamPreprocessor
        from src.surrogate.gnn.model import LayoutParamGNN

        self.torch = torch
        artifact = torch.load(Path(cfg.artifact_path), map_location="cpu", weights_only=True)
        self.pre = ParamPreprocessor.from_dict(artifact["preprocess"])
        self.y_mean = float(artifact["target_log_mean"])
        self.y_std = float(artifact["target_log_std"])
        self.model = LayoutParamGNN(**artifact["model_cfg"])
        self.model.load_state_dict(artifact["state_dict"])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        feature_df = pd.read_parquet(cfg.layout_features_path)
        match = feature_df[feature_df["layout_id"].astype(str) == self.layout_id]
        if match.empty:
            raise ValueError(
                f"Missing layout features for layout_id={self.layout_id}: {cfg.layout_features_path}"
            )
        self.layout_features = match.iloc[0][LAYOUT_FEATURES].to_dict()
        self.feasibility = self._build_feasibility_screener()

    def predict(self, decisions: list[dict[str, Any]]) -> list[float]:
        return [item.score for item in self.rank(decisions)]

    def rank(self, decisions: list[dict[str, Any]]) -> list[SteelRankingPrediction]:
        if not self.cfg.use_feasibility_penalty:
            return [
                SteelRankingPrediction(score=steel, steel_kg=steel)
                for steel in self._predict_steel(decisions)
            ]

        out: list[SteelRankingPrediction | None] = [None] * len(decisions)
        pending: list[tuple[int, dict[str, Any], tuple[Any, ...]]] = []
        for i, decision in enumerate(decisions):
            key = _decision_key(decision)
            cached = self._rank_cache.get(key)
            if cached is None:
                pending.append((i, decision, key))
            else:
                out[i] = cached

        if pending:
            items = [item[1] for item in pending]
            steel_preds = self._predict_steel(items)
            feas = self.feasibility.screen(items) if self.feasibility is not None else []
            for (idx, _, key), steel, decision in zip(pending, steel_preds, feas):
                prob = float(decision.probability)
                risk = max(0.0, min(1.0, 1.0 - prob))
                score = float(steel) + float(self.cfg.feasibility_penalty_kg) * risk
                item = SteelRankingPrediction(
                    score=score,
                    steel_kg=float(steel),
                    feasible_probability=prob,
                    infeasible_risk=risk,
                )
                self._rank_cache[key] = item
                out[idx] = item

        return [item for item in out if item is not None]

    def _predict_steel(self, decisions: list[dict[str, Any]]) -> list[float]:
        out: list[float | None] = [None] * len(decisions)
        pending: list[tuple[int, dict[str, Any], tuple[Any, ...]]] = []
        for i, decision in enumerate(decisions):
            key = _decision_key(decision)
            cached = self._cache.get(key)
            if cached is None:
                pending.append((i, decision, key))
            else:
                out[i] = cached

        if pending:
            df = self._build_frame([item[1] for item in pending])
            preds = self._predict_df(df)
            for (idx, _, key), pred in zip(pending, preds):
                self._cache[key] = float(pred)
                out[idx] = float(pred)

        return [float(v) for v in out if v is not None]

    def _build_feasibility_screener(self) -> GNNFeasibilityScreener | None:
        if not self.cfg.use_feasibility_penalty:
            return None
        cfg = SurrogateScreeningConfig(
            enabled=True,
            artifact_path=self.cfg.feasibility_artifact_path,
            graph_cache_dir=self.cfg.feasibility_graph_cache_dir,
            layout_features_path=self.cfg.feasibility_layout_features_path,
            screening_threshold=self.cfg.feasibility_threshold,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
        )
        return GNNFeasibilityScreener(
            cfg,
            layout_id=self.layout_id,
            fixed_params=self.fixed_params,
        )

    def _build_frame(self, decisions: list[dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for i, decision in enumerate(decisions):
            row = {
                "layout_id": self.layout_id,
                "sample_id": f"opt_rank_{i}",
                **self.fixed_params,
                **decision,
                **self.layout_features,
            }
            row.setdefault("conc_mid", row["conc_bot"])
            row.setdefault("conc_top", row["conc_bot"])
            rows.append(row)
        return pd.DataFrame(rows)

    def _predict_df(self, df: pd.DataFrame) -> list[float]:
        from torch_geometric.loader import DataLoader

        from src.surrogate.gnn.dataset import SurrogateGNNDataset

        ds = SurrogateGNNDataset(df, graph_cache_dir=self.cfg.graph_cache_dir, preprocessor=self.pre)
        loader = DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        preds = []
        with self.torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                pred_norm = self.model(batch)
                pred = self.torch.expm1(pred_norm * self.y_std + self.y_mean)
                preds.extend(pred.detach().cpu().tolist())
        return [float(max(v, 0.0)) for v in preds]


def _decision_key(decision: dict[str, Any]) -> tuple[Any, ...]:
    return tuple((name, decision[name]) for name in sorted(decision.keys()))
