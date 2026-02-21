from __future__ import annotations

import time
from dataclasses import dataclass
import pandas as pd


@dataclass
class StreamMetrics:
    latency_ms: float
    throughput_rows_per_s: float


def stream_dataframe(df: pd.DataFrame, chunk_size: int):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size]


def process_stream(df: pd.DataFrame, chunk_size: int, process_fn) -> tuple[pd.DataFrame, StreamMetrics]:
    start = time.perf_counter()
    outputs = []
    processed = 0
    for chunk in stream_dataframe(df, chunk_size):
        outputs.append(process_fn(chunk))
        processed += len(chunk)
    elapsed = time.perf_counter() - start
    latency_ms = (elapsed / max(1, processed)) * 1000
    throughput = processed / max(elapsed, 1e-6)
    return pd.concat(outputs, ignore_index=True), StreamMetrics(latency_ms=latency_ms, throughput_rows_per_s=throughput)


def compare_batch_vs_streaming(df: pd.DataFrame, process_fn, chunk_size: int) -> dict[str, float]:
    t0 = time.perf_counter()
    batch_out = process_fn(df)
    batch_elapsed = time.perf_counter() - t0

    stream_out, stream_metrics = process_stream(df, chunk_size=chunk_size, process_fn=process_fn)
    assert len(batch_out) == len(stream_out)

    return {
        "batch_time_s": batch_elapsed,
        "stream_time_s": len(df) / stream_metrics.throughput_rows_per_s,
        "stream_latency_ms_per_row": stream_metrics.latency_ms,
        "stream_throughput_rows_per_s": stream_metrics.throughput_rows_per_s,
    }
