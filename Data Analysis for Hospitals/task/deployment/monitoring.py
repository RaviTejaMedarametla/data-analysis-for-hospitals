from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd


def build_monitoring_summary(
    alert_flags: pd.Series,
    risk_probabilities: pd.Series,
    stream_latency_ms_per_row: float,
) -> dict[str, float | int | str | None]:
    alert_count = int(alert_flags.sum())
    total = len(alert_flags)
    first_alert_index = int(alert_flags.idxmax()) if alert_count > 0 else None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "samples_observed": total,
        "alert_count": alert_count,
        "alert_rate": float(alert_count / max(total, 1)),
        "mean_risk_probability": float(risk_probabilities.mean()),
        "risk_probability_std": float(risk_probabilities.std(ddof=0)),
        "estimated_data_drift": float(abs(risk_probabilities.mean() - 0.5)),
        "stream_latency_ms_per_row": float(stream_latency_ms_per_row),
        "first_alert_index": first_alert_index,
    }
