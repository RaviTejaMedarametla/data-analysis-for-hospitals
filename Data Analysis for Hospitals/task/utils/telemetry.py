from __future__ import annotations

import os
import time
from pathlib import Path


def _read_rapl_uj() -> float | None:
    base = Path('/sys/class/powercap')
    if not base.exists():
        return None
    for p in base.glob('intel-rapl*/energy_uj'):
        try:
            return float(p.read_text().strip())
        except Exception:
            continue
    return None


def capture_resource_snapshot() -> dict[str, float]:
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        cpu_pct = psutil.cpu_percent(interval=0.0)
        return {"cpu_percent": float(cpu_pct), "memory_rss_mb": float(mem_mb), "psutil_available": 1.0}
    except Exception:
        return {"cpu_percent": float('nan'), "memory_rss_mb": float('nan'), "psutil_available": 0.0}


def measure_block_energy_runtime(run_fn) -> tuple[dict, dict[str, float]]:
    before_rapl = _read_rapl_uj()
    before = capture_resource_snapshot()
    start = time.perf_counter()
    out = run_fn()
    elapsed_s = time.perf_counter() - start
    after = capture_resource_snapshot()
    after_rapl = _read_rapl_uj()

    measured_j = float('nan')
    if before_rapl is not None and after_rapl is not None and after_rapl >= before_rapl:
        measured_j = (after_rapl - before_rapl) / 1e6

    telemetry = {
        "runtime_s": float(elapsed_s),
        "cpu_percent_before": before["cpu_percent"],
        "cpu_percent_after": after["cpu_percent"],
        "memory_rss_mb_before": before["memory_rss_mb"],
        "memory_rss_mb_after": after["memory_rss_mb"],
        "measured_energy_joules": measured_j,
        "rapl_available": 0.0 if before_rapl is None else 1.0,
        "psutil_available": before["psutil_available"],
    }
    return out, telemetry
