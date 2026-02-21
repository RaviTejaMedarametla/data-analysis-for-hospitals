from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
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


def simulate_async_stream(
    df: pd.DataFrame,
    process_fn,
    base_arrival_ms: float,
    jitter_ms: float,
    processing_rate_rows_per_s: float,
    max_buffer: int,
    random_state: int,
) -> dict[str, float]:
    rng = np.random.default_rng(random_state)
    queue = 0
    dropped = 0
    max_queue = 0
    simulated_time_s = 0.0
    total_processed = 0

    for _ in range(len(df)):
        arrival = max(0.1, base_arrival_ms + rng.normal(0, jitter_ms)) / 1000.0
        simulated_time_s += arrival
        queue += 1

        capacity = max(1, int(processing_rate_rows_per_s * arrival))
        processed_now = min(queue, capacity)
        queue -= processed_now
        total_processed += processed_now

        if queue > max_buffer:
            dropped += queue - max_buffer
            queue = max_buffer
        max_queue = max(max_queue, queue)

    if queue > 0:
        flush_t = queue / max(processing_rate_rows_per_s, 1e-6)
        simulated_time_s += flush_t
        total_processed += queue
        queue = 0

    processed_df = process_fn(df.iloc[:total_processed])
    throughput = total_processed / max(simulated_time_s, 1e-6)
    return {
        "async_sim_time_s": float(simulated_time_s),
        "async_processed_rows": float(total_processed),
        "async_dropped_rows": float(dropped),
        "async_max_queue_depth": float(max_queue),
        "async_throughput_rows_per_s": float(throughput),
        "async_output_rows": float(len(processed_df)),
    }


def compare_batch_vs_streaming(df: pd.DataFrame, process_fn, chunk_size: int) -> dict[str, float]:
    t0 = time.perf_counter()
    batch_out = process_fn(df)
    batch_elapsed = time.perf_counter() - t0

    stream_out, stream_metrics = process_stream(df, chunk_size=chunk_size, process_fn=process_fn)
    assert len(batch_out) == len(stream_out)

    async_stats = simulate_async_stream(
        df=df,
        process_fn=process_fn,
        base_arrival_ms=8.0,
        jitter_ms=3.0,
        processing_rate_rows_per_s=150.0,
        max_buffer=64,
        random_state=42,
    )

    return {
        "batch_time_s": batch_elapsed,
        "stream_time_s": len(df) / stream_metrics.throughput_rows_per_s,
        "stream_latency_ms_per_row": stream_metrics.latency_ms,
        "stream_throughput_rows_per_s": stream_metrics.throughput_rows_per_s,
        **async_stats,
    }
