from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import psutil


def run_onnx_inference(
    model_path: Path,
    X: np.ndarray,
    warmup_runs: int = 3,
    iterations: int = 50,
    output_json: Path | None = None,
) -> dict[str, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        result = {"error": f"onnxruntime_unavailable: {exc}"}
        if output_json:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(result, indent=2))
        return result

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rss_before = psutil.Process().memory_info().rss

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: X.astype(np.float32)})

    times_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: X.astype(np.float32)})
        times_ms.append((time.perf_counter() - start) * 1000)

    rss_after = psutil.Process().memory_info().rss
    result = {
        "mean_latency_ms": float(np.mean(times_ms)),
        "std_latency_ms": float(np.std(times_ms)),
        "p95_latency_ms": float(np.percentile(times_ms, 95)),
        "rss_before_mb": float(rss_before / (1024 ** 2)),
        "rss_after_mb": float(rss_after / (1024 ** 2)),
        "rss_delta_mb": float((rss_after - rss_before) / (1024 ** 2)),
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2))
    return result
