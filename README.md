# Hospital Analytics Pipeline

## Overview
This repository implements a hardware-aware machine learning pipeline for tabular hospital analytics with deterministic execution characteristics.
The project is maintained as part of a broader AI systems engineering portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

## System Architecture
The pipeline is organized in modular stages under `Data Analysis for Hospitals/task/`:

- `ingestion/`: dataset loading and manifest creation.
- `preprocessing/`: schema validation and data quality normalization.
- `feature_engineering/`: derived feature construction for downstream models.
- `modeling/`: predictive modeling workflows.
- `anomaly_detection/`: anomaly scoring and early warning simulation.
- `real_time/`: streaming inference utilities.
- `deployment/`: CPU inference, ONNX export, and monitoring summaries.
- `evaluation/`: benchmarking and metric reporting.
- `utils/`: reproducibility and runtime support utilities.

## Features
- End-to-end CLI-driven analytics workflow for manifest generation, pipeline execution, and early warning experiments.
- Deterministic runtime controls for reproducible training and evaluation behavior.
- Hardware-aware benchmark outputs covering latency, memory, and deployment-oriented diagnostics.
- Artifact generation for experiment tracking and operational review.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
cd "Data Analysis for Hospitals/task"
python cli.py manifest
python cli.py run
python cli.py early-warning-experiment
```

Generated artifacts are written to `Data Analysis for Hospitals/task/artifacts/`.

## Reproducibility
- Use pinned dependencies from `requirements.txt`.
- Execute the CLI commands from the same working directory and input datasets.
- Run repository tests to verify deterministic behavior and environment compatibility.

## Related Projects
This repository is part of the same AI systems portfolio as:

- `neural-network-systems`
- `digit-classification-benchmark`
- `edge-ai-model-optimization`
- `hospital-analytics-pipeline`
- `nba-data-engineering`
- `ai-systems-ml-platform`

> Naming recommendation: consider renaming this repository on GitHub from `data-analysis-for-hospitals` to `hospital-analytics-pipeline` for portfolio consistency.
