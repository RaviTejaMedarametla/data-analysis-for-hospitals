from __future__ import annotations

import numpy as np
import pandas as pd


class OutlierDetector:
    def __init__(self, random_state: int = 42):
        self.center: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "OutlierDetector":
        vals = X.values.astype(float)
        self.center = vals.mean(axis=0)
        self.scale = vals.std(axis=0) + 1e-6
        return self

    def score_samples(self, X: pd.DataFrame) -> pd.Series:
        z = np.abs((X.values.astype(float) - self.center) / self.scale)
        score = z.mean(axis=1)
        return pd.Series(score, index=X.index, name="anomaly_score")

    def detect(self, X: pd.DataFrame, threshold_quantile: float = 0.9) -> pd.DataFrame:
        scores = self.score_samples(X)
        threshold = scores.quantile(threshold_quantile)
        return pd.DataFrame({"anomaly_score": scores, "is_anomaly": scores >= threshold})
