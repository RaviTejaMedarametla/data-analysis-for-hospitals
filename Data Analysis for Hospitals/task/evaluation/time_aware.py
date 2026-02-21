from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_origin_splits(n_samples: int, min_train: int, horizon: int, step: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    train_end = min_train
    while train_end + horizon <= n_samples:
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, train_end + horizon)
        splits.append((train_idx, test_idx))
        train_end += step
    return splits


def detection_delay_distribution(scores: pd.Series, events: pd.Series, timestamps: pd.DatetimeIndex, threshold: float) -> dict[str, float]:
    alert_idx = np.where(scores.values >= threshold)[0]
    event_idx = np.where(events.values == 1)[0]
    delays = []
    lead_times = []
    for ei in event_idx:
        prior_alerts = alert_idx[alert_idx <= ei]
        post_alerts = alert_idx[alert_idx >= ei]
        if len(prior_alerts) > 0:
            lead_times.append(float((timestamps[ei] - timestamps[prior_alerts[-1]]).total_seconds()))
        if len(post_alerts) > 0:
            delays.append(float((timestamps[post_alerts[0]] - timestamps[ei]).total_seconds()))
    return {
        "event_count": float(len(event_idx)),
        "median_detection_delay_s": float(np.median(delays)) if delays else float("inf"),
        "p90_detection_delay_s": float(np.quantile(delays, 0.9)) if delays else float("inf"),
        "median_early_warning_gain_s": float(np.median(lead_times)) if lead_times else 0.0,
    }


def false_alarms_per_hour(alerts: pd.Series, events: pd.Series, timestamps: pd.DatetimeIndex) -> float:
    false_alerts = float(((alerts == 1) & (events == 0)).sum())
    duration_s = max((timestamps[-1] - timestamps[0]).total_seconds(), 1.0)
    return false_alerts / (duration_s / 3600.0)


def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    thresholds = np.unique(y_score)[::-1]
    precision = []
    recall = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
    return np.array(precision, dtype=float), np.array(recall, dtype=float)


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    return float(np.trapz(y[order], x[order]))


def precision_recall_at_budget(scores: pd.Series, events: pd.Series, budget_fractions: list[float]) -> list[dict[str, float]]:
    y_true = events.values.astype(int)
    y_score = scores.values.astype(float)
    precision, recall = _precision_recall_curve(y_true, y_score)
    pr_auc = _auc(recall, precision)

    rows = []
    for frac in budget_fractions:
        cutoff = np.quantile(y_score, 1 - frac)
        pred = (y_score >= cutoff).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        p = tp / max(tp + fp, 1.0)
        r = tp / max(tp + fn, 1.0)
        rows.append({"alert_budget_fraction": float(frac), "precision": p, "recall": r, "pr_auc": pr_auc})
    return rows
