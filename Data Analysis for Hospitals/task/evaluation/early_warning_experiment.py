from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from anomaly_detection.detectors import OutlierDetector
from anomaly_detection.early_warning import evaluate_detection_latency
from utils.hardware import HardwareProfile, auto_adjust_batch_size, compute_utilization


@dataclass(frozen=True)
class ConstraintScenario:
    memory_limit_mb: int
    compute_budget: int
    stream_interval_ms: int


def _simulate_scenario(
    df: pd.DataFrame,
    y_true: pd.Series,
    feature_cols: list[str],
    scenario: ConstraintScenario,
) -> dict[str, float]:
    profile = HardwareProfile(memory_limit_mb=scenario.memory_limit_mb, compute_budget=scenario.compute_budget)
    effective_batch = auto_adjust_batch_size(initial_batch=64, feature_count=len(feature_cols), profile=profile)

    detector = OutlierDetector(random_state=42).fit(df[feature_cols])
    anomaly_frame = detector.detect(df[feature_cols], threshold_quantile=0.9)

    scores = anomaly_frame["anomaly_score"]
    threshold = float(scores.quantile(0.9))
    predicted_events = (scores >= threshold).astype(int)
    timestamps = pd.date_range("2025-01-01", periods=len(df), freq=f"{scenario.stream_interval_ms}ms")

    latency_s = evaluate_detection_latency(scores, y_true, timestamps)
    accuracy = float((predicted_events.values == y_true.values).mean())
    fp = float(((predicted_events == 1) & (y_true == 0)).sum())
    fp_rate = fp / max(float((y_true == 0).sum()), 1.0)

    ops = len(df) * len(feature_cols)
    utilization = compute_utilization(operations=ops, profile=profile)

    quality = max(0.0, accuracy - 0.5 * fp_rate)

    return {
        "memory_limit_mb": float(scenario.memory_limit_mb),
        "compute_budget": float(scenario.compute_budget),
        "stream_interval_ms": float(scenario.stream_interval_ms),
        "effective_batch_size": float(effective_batch),
        "detection_latency_s": float(latency_s),
        "prediction_accuracy": accuracy,
        "false_positives": fp,
        "false_positive_rate": fp_rate,
        "compute_utilization": utilization,
        "detection_quality": quality,
    }


def run_hardware_constrained_early_warning_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    scenarios: Iterable[ConstraintScenario],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    y_true = df[target_col].isin(["appendicitis", "pregnancy"]).astype(int)
    rows = [_simulate_scenario(df, y_true, feature_cols, scenario) for scenario in scenarios]
    result_df = pd.DataFrame(rows).sort_values(["memory_limit_mb", "compute_budget", "stream_interval_ms"])

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "early_warning_hardware_experiment.csv"
    result_df.to_csv(csv_path, index=False)

    fig1 = output_dir / "latency_vs_accuracy.png"
    plt.figure(figsize=(7, 5))
    plt.scatter(result_df["detection_latency_s"], result_df["prediction_accuracy"], c=result_df["compute_budget"], cmap="viridis")
    plt.xlabel("Detection latency (s)")
    plt.ylabel("Prediction accuracy")
    plt.title("Latency vs Accuracy under Hardware Constraints")
    plt.colorbar(label="Compute budget")
    plt.tight_layout()
    plt.savefig(fig1)
    plt.close()

    fig2 = output_dir / "resource_vs_detection_quality.png"
    resource_score = result_df["memory_limit_mb"] * result_df["compute_budget"] / result_df["stream_interval_ms"]
    plt.figure(figsize=(7, 5))
    plt.scatter(resource_score, result_df["detection_quality"], c=result_df["false_positive_rate"], cmap="magma_r")
    plt.xlabel("Resource score (memory*compute/stream_interval)")
    plt.ylabel("Detection quality (accuracy - 0.5*FPR)")
    plt.title("Resource vs Detection Quality")
    plt.colorbar(label="False positive rate")
    plt.tight_layout()
    plt.savefig(fig2)
    plt.close()

    artifacts = {
        "results_csv": str(csv_path),
        "latency_vs_accuracy_plot": str(fig1),
        "resource_vs_detection_quality_plot": str(fig2),
    }
    return result_df, artifacts


def summarize_experiment(df: pd.DataFrame) -> dict[str, float]:
    return {
        "latency_mean_s": float(df["detection_latency_s"].mean()),
        "latency_std_s": float(df["detection_latency_s"].std(ddof=0)),
        "accuracy_mean": float(df["prediction_accuracy"].mean()),
        "accuracy_std": float(df["prediction_accuracy"].std(ddof=0)),
        "false_positive_rate_mean": float(df["false_positive_rate"].mean()),
        "false_positive_rate_std": float(df["false_positive_rate"].std(ddof=0)),
        "quality_mean": float(df["detection_quality"].mean()),
        "quality_std": float(df["detection_quality"].std(ddof=0)),
    }
