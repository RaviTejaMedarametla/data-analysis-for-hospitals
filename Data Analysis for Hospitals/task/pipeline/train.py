from __future__ import annotations

from config import CONFIG
from feature_engineering.features import build_features
from ingestion.loader import load_hospital_data, merge_hospital_data
from modeling.predictive import train_predictive_models
from preprocessing.cleaning import clean_hospital_data


def run_training_pipeline():
    datasets = load_hospital_data(CONFIG.data_dir)
    merged = merge_hospital_data(datasets)
    clean = clean_hospital_data(merged)
    feat = build_features(clean)
    artifacts = train_predictive_models(
        feat,
        CONFIG.feature_columns,
        CONFIG.target_risk,
        CONFIG.target_outcome,
        split_seed=CONFIG.split_seed,
    )
    return feat, artifacts
