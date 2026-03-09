from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psutil

from deployment.onnx_inference import run_onnx_inference


def benchmark_cpu(model, X, onnx_model_path: Path, output_json: Path) -> dict:
    process = psutil.Process()
    rss_before = process.memory_info().rss
    start = time.perf_counter()
    _ = model.predict_proba(X)
    baseline_ms = (time.perf_counter() - start) * 1000
    rss_after = process.memory_info().rss

    onnx_metrics = run_onnx_inference(onnx_model_path, X.values.astype(np.float32))
    if "error" in onnx_metrics:
        speedup = 0.0
        memory_savings_mb = 0.0
    else:
        speedup = baseline_ms / onnx_metrics["mean_latency_ms"] if onnx_metrics["mean_latency_ms"] > 0 else 0.0
        memory_savings_mb = ((rss_after - rss_before) / (1024 ** 2)) - onnx_metrics["rss_delta_mb"]

    result = {
        "baseline_latency_ms": float(baseline_ms),
        "baseline_rss_delta_mb": float((rss_after - rss_before) / (1024 ** 2)),
        "onnx": onnx_metrics,
        "speedup_x": float(speedup),
        "memory_savings_mb": float(memory_savings_mb),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2))
    return result
