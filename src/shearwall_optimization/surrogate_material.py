from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.surrogate.features.consts import LAYOUT_FEATURES, PARAM_FEATURES

from .cost_estimation import estimate_concrete_kg

CAT_COLS = ["conc_bot", "site_class", "intensity", "seismic_group"]


@dataclass(frozen=True)
class SteelMaterialPrediction:
    steel_mean_kg: float
    steel_upper_kg: float
    concrete_kg: float
    steel_rel_upper_gap: float


class SteelMaterialPredictor:
    def __init__(self, artifact_dir: str | Path):
        artifact = Path(artifact_dir)
        self.mean_model = joblib.load(artifact / "steel_mean_lightgbm.joblib")
        self.upper_model = joblib.load(artifact / "steel_upper_lightgbm.joblib")
        self.feature_columns = json.loads((artifact / "feature_columns.json").read_text(encoding="utf-8"))

        metrics_path = artifact / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        self.upper_margin_kg = float(metrics.get("calibration", {}).get("upper_margin_kg", 0.0))

    def predict(self, df: pd.DataFrame) -> list[SteelMaterialPrediction]:
        x = self._encode(df)
        steel_mean = self.mean_model.predict(x)
        steel_upper = self.upper_model.predict(x) + self.upper_margin_kg
        steel_upper = [max(float(u), float(m)) for m, u in zip(steel_mean, steel_upper)]

        out = []
        for row, mean, upper in zip(df.to_dict("records"), steel_mean, steel_upper):
            concrete = estimate_concrete_kg(row)
            gap = (float(upper) - float(mean)) / max(float(mean), 1.0)
            out.append(
                SteelMaterialPrediction(
                    steel_mean_kg=float(mean),
                    steel_upper_kg=float(upper),
                    concrete_kg=float(concrete),
                    steel_rel_upper_gap=float(gap),
                )
            )
        return out

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in PARAM_FEATURES + LAYOUT_FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Steel material surrogate input missing columns: {missing}")

        x = df[PARAM_FEATURES + LAYOUT_FEATURES].copy()
        for col in CAT_COLS:
            x[col] = x[col].astype(str)
        x = pd.get_dummies(x, columns=CAT_COLS, dtype=float)
        return x.reindex(columns=self.feature_columns, fill_value=0.0)
