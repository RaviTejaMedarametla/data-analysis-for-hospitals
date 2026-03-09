from __future__ import annotations

from pathlib import Path
import pandas as pd

from config import CONFIG
from feature_engineering.features import build_features
from modeling.predictive import train_predictive_models, evaluate_predictive_models


def run_ablation_studies(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    studies = {
        "all_features": CONFIG.feature_columns,
        "no_body_metrics": ["age", "children", "months"],
        "vitals_only": ["height", "weight", "bmi"],
    }
    rows = []
    feat_df = build_features(df)
    for name, cols in studies.items():
        artifacts = train_predictive_models(
            feat_df,
            cols,
            CONFIG.target_risk,
            CONFIG.target_outcome,
            split_seed=CONFIG.split_seed,
        )
        metrics = evaluate_predictive_models(artifacts)
        rows.append({"study": name, **metrics, "feature_count": len(cols)})
    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "ablations.csv", index=False)
    return out
