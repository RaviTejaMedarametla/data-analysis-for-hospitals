from __future__ import annotations

import argparse
import json

from config import CONFIG
from ingestion.versioning import create_dataset_manifest
from pipeline.run import run_pipeline
from pipeline.train import run_training_pipeline
from experiments.ablations import run_ablation_studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare AI research pipeline")
    parser.add_argument("command", choices=["run", "manifest", "ablation"], help="Pipeline command")
    args = parser.parse_args()

    if args.command == "manifest":
        manifest = create_dataset_manifest(CONFIG.data_dir, CONFIG.output_dir / "dataset_manifest.json")
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "ablation":
        feat, _ = run_training_pipeline()
        frame = run_ablation_studies(feat, CONFIG.results_dir)
        print(frame.to_json(orient="records", indent=2))
        return

    print(json.dumps(run_pipeline(), indent=2, default=float))


if __name__ == "__main__":
    main()
