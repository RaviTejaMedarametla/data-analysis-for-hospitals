from __future__ import annotations

from pathlib import Path

import pandas as pd


def _estimate_layer_costs(feature_count: int, batch_size: int) -> pd.DataFrame:
    rows = [
        {"operator": "input_normalization", "latency_ms": 0.01 * batch_size, "memory_kb": feature_count * batch_size * 8 / 1024},
        {"operator": "linear_projection", "latency_ms": 0.03 * batch_size, "memory_kb": feature_count * batch_size * 12 / 1024},
        {"operator": "activation", "latency_ms": 0.008 * batch_size, "memory_kb": feature_count * batch_size * 4 / 1024},
        {"operator": "decision_head", "latency_ms": 0.012 * batch_size, "memory_kb": batch_size * 8 / 1024},
    ]
    return pd.DataFrame(rows)


def build_hardware_profile_table(feature_count: int, batch_size: int, stream_interval_ms: int) -> dict:
    op_df = _estimate_layer_costs(feature_count, batch_size)
    total_latency_ms = float(op_df["latency_ms"].sum())
    total_memory_kb = float(op_df["memory_kb"].sum())

    bytes_moved = total_memory_kb * 1024
    bandwidth_mb_s = (bytes_moved / (1024 * 1024)) / max(total_latency_ms / 1000, 1e-9)
    utilization = min(1.0, total_latency_ms / max(stream_interval_ms, 1))

    return {
        "operator_profile": op_df.to_dict(orient="records"),
        "totals": {
            "latency_ms": total_latency_ms,
            "memory_kb": total_memory_kb,
            "estimated_bandwidth_mb_s": float(bandwidth_mb_s),
            "stream_utilization": float(utilization),
        },
        "precision_tradeoffs": {
            "fp32_memory_kb": total_memory_kb,
            "fp16_memory_kb": total_memory_kb * 0.5,
            "int8_memory_kb": total_memory_kb * 0.25,
            "note": "Latency effects from quantization are deployment dependent and should be validated on target hardware.",
        },
        "edge_constraints": {
            "cache_sensitivity": "Small stream chunks reduce cache reuse and can increase per-record overhead.",
            "bottleneck": "Memory movement dominates when feature width increases without proportional compute budget growth.",
        },
    }


def write_hardware_profile_artifacts(profile: dict, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    op_df = pd.DataFrame(profile["operator_profile"])
    totals_df = pd.DataFrame([profile["totals"]])

    operator_path = output_dir / "operator_profile.csv"
    totals_path = output_dir / "hardware_totals.csv"

    op_df.to_csv(operator_path, index=False)
    totals_df.to_csv(totals_path, index=False)
    return {"operator_profile_csv": str(operator_path), "hardware_totals_csv": str(totals_path)}
