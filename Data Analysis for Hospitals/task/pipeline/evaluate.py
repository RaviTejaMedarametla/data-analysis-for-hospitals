from __future__ import annotations

from config import CONFIG
from evaluation.benchmark import run_repeated_benchmark
from modeling.predictive import evaluate_predictive_models


def run_evaluation_pipeline(artifacts):
    metrics = evaluate_predictive_models(artifacts)
    bench = run_repeated_benchmark(
        lambda: evaluate_predictive_models(artifacts),
        metric_key="risk_accuracy",
        runs=CONFIG.benchmark_runs,
        confidence=CONFIG.confidence_level,
    )
    return metrics, bench.__dict__
