from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log_experiment(payload: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    if output_path.exists():
        history = json.loads(output_path.read_text())
    else:
        history = []
    history.append(record)
    output_path.write_text(json.dumps(history, indent=2))
