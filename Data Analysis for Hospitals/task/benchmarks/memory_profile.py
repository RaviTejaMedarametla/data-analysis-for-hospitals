from __future__ import annotations

import json
from pathlib import Path

import psutil


def profile_memory(fn, output_path: Path) -> dict:
    process = psutil.Process()
    before = process.memory_info().rss
    fn()
    after = process.memory_info().rss
    result = {
        "rss_before_mb": float(before / (1024 ** 2)),
        "rss_after_mb": float(after / (1024 ** 2)),
        "rss_delta_mb": float((after - before) / (1024 ** 2)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result
