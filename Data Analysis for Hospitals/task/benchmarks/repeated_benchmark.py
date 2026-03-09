from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


def run_repeated_benchmark(fn, output_path: Path, warmup_runs: int = 3, runs: int = 30) -> dict:
    for _ in range(warmup_runs):
        fn()
    times = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        times.append((time.perf_counter() - s) * 1000)
    result = {
        "runs": runs,
        "warmup_runs": warmup_runs,
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result
