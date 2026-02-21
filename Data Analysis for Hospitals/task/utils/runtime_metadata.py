from __future__ import annotations

import json
import platform
import sys
from pathlib import Path


def collect_runtime_metadata() -> dict:
    packages = {}
    try:
        import importlib.metadata as md

        for dist in md.distributions():
            name = dist.metadata.get('Name')
            if name:
                packages[name] = dist.version
    except Exception:
        pass

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "packages": dict(sorted(packages.items())),
    }


def write_runtime_metadata(path: Path) -> str:
    payload = collect_runtime_metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return str(path)
