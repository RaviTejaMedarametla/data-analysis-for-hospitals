from __future__ import annotations

import json
import time
from pathlib import Path


def profile_latency(stages: dict[str, callable], output_path: Path) -> dict:
    out = {}
    for name, fn in stages.items():
        start = time.perf_counter()
        fn()
        out[name] = {"latency_ms": (time.perf_counter() - start) * 1000}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return out
