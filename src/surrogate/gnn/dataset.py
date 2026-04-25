from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.surrogate.training.lightgbm_baseline import CLASS_TASK, LAYOUT_FEATURES, PARAM_FEATURES

CAT_COLS = ["conc_bot", "intensity", "site_class", "seismic_group"]
FEATURE_COLS = PARAM_FEATURES + LAYOUT_FEATURES
NUM_COLS = [col for col in FEATURE_COLS if col not in CAT_COLS]


@dataclass(frozen=True)
class GNNDataConfig:
    dataset_path: str = r"data\parametric\surrogate_dataset\splits\surrogate_samples_with_splits.parquet"
    graph_cache_dir: str = r"data\parametric\surrogate_dataset\gnn_graph_cache"
    batch_size: int = 128
    num_workers: int = 0


class ParamPreprocessor:
    def __init__(self, num_cols: list[str], cat_cols: list[str]):
        self.num_cols = list(num_cols)
        self.cat_cols = list(cat_cols)
        self.num_mean: dict[str, float] = {}
        self.num_std: dict[str, float] = {}
        self.cat_vocab: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> None:
        num = df[self.num_cols].copy()
        self.num_mean = {c: float(v) for c, v in num.mean(axis=0).items()}
        self.num_std = {c: float(v if float(v) != 0.0 else 1.0) for c, v in num.std(axis=0, ddof=0).items()}

        for col in self.cat_cols:
            raw = df[col].astype(str).fillna("<unk>")
            unique_vals = sorted(v for v in raw.unique() if v != "<unk>")
            vocab = {"<unk>": 0}
            for i, val in enumerate(unique_vals, start=1):
                vocab[val] = i
            self.cat_vocab[col] = vocab

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x_num = df[self.num_cols].copy()
        mean = pd.Series(self.num_mean, dtype=float)
        std = pd.Series(self.num_std, dtype=float)
        x_num = x_num.fillna(mean)
        x_num = ((x_num - mean) / std).to_numpy(dtype=np.float32)

        cat_arrays: list[np.ndarray] = []
        for col in self.cat_cols:
            vocab = self.cat_vocab[col]
            raw = df[col].astype(str).fillna("<unk>")
            cat_arrays.append(raw.map(lambda v: vocab.get(v, 0)).to_numpy(dtype=np.int64))
        x_cat = np.stack(cat_arrays, axis=1)
        return x_num, x_cat

    def export(self) -> dict[str, object]:
        return {
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "num_mean": [self.num_mean[c] for c in self.num_cols],
            "num_std": [self.num_std[c] for c in self.num_cols],
            "cat_vocab": self.cat_vocab,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ParamPreprocessor":
        obj = cls(list(payload["num_cols"]), list(payload["cat_cols"]))
        obj.num_mean = {c: float(v) for c, v in zip(obj.num_cols, payload["num_mean"])}
        obj.num_std = {c: float(v) for c, v in zip(obj.num_cols, payload["num_std"])}
        obj.cat_vocab = {k: dict(v) for k, v in dict(payload["cat_vocab"]).items()}
        return obj


class SurrogateGNNDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        graph_cache_dir: str | Path,
        preprocessor: ParamPreprocessor,
    ):
        self.df = df.reset_index(drop=True)
        self.graph_cache_dir = Path(graph_cache_dir)
        self.preprocessor = preprocessor

        self.x_num, self.x_cat = preprocessor.transform(self.df)
        self.y = self.df[CLASS_TASK].astype(int).to_numpy(dtype=np.float32)

        self._graph_cache: dict[str, dict[str, torch.Tensor]] = {}
        layout_ids = sorted(self.df["layout_id"].astype(str).unique().tolist())
        for layout_id in layout_ids:
            path = self.graph_cache_dir / f"{layout_id}.pt"
            if not path.exists():
                raise ValueError(f"Missing graph cache for layout {layout_id}: {path}")
            self._graph_cache[layout_id] = torch.load(path, map_location="cpu", weights_only=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Data:
        row = self.df.iloc[idx]
        layout_id = str(row["layout_id"])
        g = self._graph_cache[layout_id]

        data = Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            graph_feat=g["graph_feat"],
            p_num=torch.tensor(self.x_num[idx][None, :], dtype=torch.float32),
            p_cat=torch.tensor(self.x_cat[idx][None, :], dtype=torch.long),
            y=torch.tensor([self.y[idx]], dtype=torch.float32),
            sample_id=str(row.get("sample_id", "")),
            layout_id=layout_id,
        )
        return data


def build_dataloaders(
    cfg: GNNDataConfig,
    preprocess_from: dict[str, object] | None = None,
) -> tuple[dict[str, DataLoader], ParamPreprocessor, tuple[int, int, int]]:
    df = pd.read_parquet(cfg.dataset_path)
    required = ["split", "layout_id", CLASS_TASK] + FEATURE_COLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("train/val/test split is empty")

    if preprocess_from is None:
        pre = ParamPreprocessor(NUM_COLS, CAT_COLS)
        pre.fit(train_df)
    else:
        pre = ParamPreprocessor.from_dict(preprocess_from)

    ds_train = SurrogateGNNDataset(train_df, cfg.graph_cache_dir, pre)
    ds_val = SurrogateGNNDataset(val_df, cfg.graph_cache_dir, pre)
    ds_test = SurrogateGNNDataset(test_df, cfg.graph_cache_dir, pre)

    loaders = {
        "train": DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
        "val": DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
        "test": DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers),
    }

    sample = ds_train[0]
    dims = (int(sample.x.shape[1]), int(sample.edge_attr.shape[1]), int(sample.graph_feat.shape[1]))
    return loaders, pre, dims
