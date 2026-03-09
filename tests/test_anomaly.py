import pandas as pd
from anomaly_detection.early_warning import evaluate_detection_latency


def test_latency_non_negative():
    scores = pd.Series([0.1, 0.2, 0.9, 0.95])
    events = pd.Series([0, 0, 1, 0])
    ts = pd.date_range("2024-01-01", periods=4, freq="s")
    latency = evaluate_detection_latency(scores, events, ts)
    assert latency >= 0
