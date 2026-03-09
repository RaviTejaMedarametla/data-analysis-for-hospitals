from __future__ import annotations

import pandas as pd

from anomaly_detection.detectors import OutlierDetector
from anomaly_detection.early_warning import simulate_early_warning, evaluate_detection_latency
from config import CONFIG


def run_anomaly_pipeline(feat: pd.DataFrame) -> dict:
    detector = OutlierDetector(random_state=CONFIG.random_seed).fit(feat[CONFIG.feature_columns])
    anomaly = detector.detect(feat[CONFIG.feature_columns])
    timestamps = pd.date_range("2025-01-01", periods=len(feat), freq="min")
    early_warning = simulate_early_warning(anomaly["anomaly_score"], timestamps, anomaly["anomaly_score"].quantile(0.9))
    synthetic_events = (anomaly["anomaly_score"] > anomaly["anomaly_score"].quantile(0.95)).astype(int)
    latency = evaluate_detection_latency(anomaly["anomaly_score"], synthetic_events, timestamps)
    return {"anomaly": anomaly, "alerts": early_warning, "latency": latency}
