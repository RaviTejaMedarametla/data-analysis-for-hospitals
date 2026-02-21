from __future__ import annotations

import time
from dataclasses import dataclass

from .statistics import confidence_interval


@dataclass
class BenchmarkResult:
    metric_mean: float
    metric_std: float
    metric_ci_margin: float
    latency_mean_ms: float
    latency_std_ms: float
    latency_ci_margin_ms: float


def run_repeated_benchmark(run_fn, metric_key: str, runs: int = 5) -> BenchmarkResult:
    metrics = []
    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        result = run_fn()
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics.append(float(result[metric_key]))
        latencies.append(elapsed_ms)

    m_mean, m_std, m_ci = confidence_interval(metrics)
    l_mean, l_std, l_ci = confidence_interval(latencies)
    return BenchmarkResult(m_mean, m_std, m_ci, l_mean, l_std, l_ci)


def benchmark_table_metrics(df, metric_columns: list[str]) -> dict[str, dict[str, float]]:
    summary = {}
    for col in metric_columns:
        mean, std, ci_margin = confidence_interval(df[col].astype(float).tolist())
        summary[col] = {
            "mean": mean,
            "std": std,
            "ci_margin": ci_margin,
        }
    return summary
