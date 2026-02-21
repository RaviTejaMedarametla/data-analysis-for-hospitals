from __future__ import annotations

import time
import pandas as pd


def run_cpu_inference(model, X: pd.DataFrame) -> dict[str, float]:
    start = time.perf_counter()
    probs = model.predict_proba(X)[:, 1]
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "inference_latency_ms": elapsed_ms,
        "output_mean_probability": float(probs.mean()),
        "output_std_probability": float(probs.std()),
    }
