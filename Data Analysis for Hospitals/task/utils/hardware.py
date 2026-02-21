from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HardwareProfile:
    memory_limit_mb: int
    compute_budget: int


def estimate_batch_memory_mb(batch_size: int, feature_count: int, bytes_per_feature: int = 8) -> float:
    return (batch_size * feature_count * bytes_per_feature) / (1024 * 1024)


def auto_adjust_batch_size(initial_batch: int, feature_count: int, profile: HardwareProfile) -> int:
    batch = initial_batch
    while batch > 1 and estimate_batch_memory_mb(batch, feature_count) > profile.memory_limit_mb:
        batch //= 2
    return max(1, batch)


def compute_utilization(operations: int, profile: HardwareProfile) -> float:
    return min(1.0, operations / max(profile.compute_budget, 1))
