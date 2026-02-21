from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except Exception:
    IsolationForest = None
    SKLEARN_AVAILABLE = False


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


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=contamination, random_state=random_state) if SKLEARN_AVAILABLE else None
        self.median: np.ndarray | None = None
        self.mad: np.ndarray | None = None

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        vals = X.values.astype(float)
        if self.model is not None:
            self.model.fit(vals)
        else:
            self.median = np.median(vals, axis=0)
            self.mad = np.median(np.abs(vals - self.median), axis=0) + 1e-6
        return self

    def detect(self, X: pd.DataFrame, threshold_quantile: float = 0.9) -> pd.DataFrame:
        vals = X.values.astype(float)
        if self.model is not None:
            anomaly_score = -self.model.score_samples(vals)
        else:
            robust_z = np.abs((vals - self.median) / self.mad)
            anomaly_score = robust_z.mean(axis=1)
        scores = pd.Series(anomaly_score, index=X.index, name="anomaly_score")
        threshold = scores.quantile(threshold_quantile)
        return pd.DataFrame({"anomaly_score": scores, "is_anomaly": scores >= threshold})


def compare_detectors(X: pd.DataFrame, y_true_events: pd.Series, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for name, detector in {
        "zscore": OutlierDetector(random_state=random_state),
        "isolation_forest": IsolationForestDetector(random_state=random_state),
    }.items():
        result = detector.fit(X).detect(X)
        pred = result["is_anomaly"].astype(int)
        y = y_true_events.astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        fn = float(((pred == 0) & (y == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        rows.append(
            {
                "detector": name,
                "precision": precision,
                "recall": recall,
                "anomaly_rate": float(pred.mean()),
                "score_mean": float(result["anomaly_score"].mean()),
                "sklearn_isolation_forest": float(SKLEARN_AVAILABLE),
            }
        )
    return pd.DataFrame(rows)
