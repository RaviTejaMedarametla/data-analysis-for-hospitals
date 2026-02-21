from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_early_warning(scores: pd.Series, timestamps: pd.DatetimeIndex, threshold: float) -> dict[str, float]:
    alerts = scores >= threshold
    if not alerts.any():
        return {"alert_count": 0.0, "first_alert_latency_s": float("inf")}

    first_idx = int(np.where(alerts.values)[0][0])
    latency = (timestamps[first_idx] - timestamps[0]).total_seconds()
    return {"alert_count": float(alerts.sum()), "first_alert_latency_s": float(latency)}


def evaluate_detection_latency(scores: pd.Series, ground_truth_events: pd.Series, timestamps: pd.DatetimeIndex) -> float:
    event_indices = np.where(ground_truth_events.values == 1)[0]
    if len(event_indices) == 0:
        return float("nan")
    first_event_i = event_indices[0]
    alerts = np.where(scores.values >= np.quantile(scores.values, 0.9))[0]
    post_event_alerts = alerts[alerts >= first_event_i]
    if len(post_event_alerts) == 0:
        return float("inf")
    return float((timestamps[post_event_alerts[0]] - timestamps[first_event_i]).total_seconds())
