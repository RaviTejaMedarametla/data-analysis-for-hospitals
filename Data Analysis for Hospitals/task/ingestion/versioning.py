from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def create_dataset_manifest(data_dir: Path, output_path: Path) -> dict:
    manifest = {
        "dataset_dir": str(data_dir),
        "files": [],
    }
    for path in sorted(data_dir.glob("*.csv")):
        manifest["files"].append(
            {
                "name": path.name,
                "sha256": hash_file(path),
                "size": path.stat().st_size,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    return manifest
