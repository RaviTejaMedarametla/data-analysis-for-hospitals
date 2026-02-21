from __future__ import annotations


def latency_accuracy_tradeoff(accuracy: float, latency_ms: float) -> float:
    return accuracy / max(latency_ms, 1e-6)
