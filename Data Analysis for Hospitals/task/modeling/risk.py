from __future__ import annotations

import pandas as pd


def stratify_risk(probabilities: pd.Series, low_threshold: float = 0.35, high_threshold: float = 0.7) -> pd.DataFrame:
    risk_band = pd.Series("medium", index=probabilities.index, dtype="object")
    risk_band.loc[probabilities < low_threshold] = "low"
    risk_band.loc[probabilities >= high_threshold] = "high"

    return pd.DataFrame(
        {
            "risk_probability": probabilities.astype(float),
            "risk_band": risk_band,
        }
    )


def summarize_risk_bands(risk_frame: pd.DataFrame) -> dict[str, float]:
    counts = risk_frame["risk_band"].value_counts(normalize=True)
    return {
        "low_prevalence": float(counts.get("low", 0.0)),
        "medium_prevalence": float(counts.get("medium", 0.0)),
        "high_prevalence": float(counts.get("high", 0.0)),
    }
