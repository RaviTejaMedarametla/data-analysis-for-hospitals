# Data Analysis for Hospitals

A reproducible, hardware-aware machine learning pipeline for hospital tabular analytics and early-warning evaluation.

## Overview
This repository implements an end-to-end analytics workflow for structured hospital datasets, covering ingestion, cleaning, feature engineering, predictive modeling, anomaly detection, and deployment-oriented evaluation. The system is organized as a CLI-driven pipeline that produces experiment artifacts for risk modeling, streaming inference, hardware profiling, and monitoring summaries.

The project addresses a common systems challenge in applied machine learning: model quality alone is insufficient when deployment environments are resource-constrained and operationally variable. By combining deterministic execution controls, repeatable experiment configuration, and hardware-aware benchmarking, the repository supports reliable comparisons across runs and clearer interpretation of latency, memory, and accuracy trade-offs.

## Project Motivation
Machine learning systems used in healthcare-adjacent analytics are often evaluated primarily on predictive performance, while runtime constraints and reproducibility requirements receive less attention. This repository is motivated by the need to evaluate models under practical constraints, including bounded compute budgets, memory limits, and streaming workloads.

The project also emphasizes deterministic experimentation and traceable artifacts. Reproducible pipelines are essential for validating model behavior, comparing architectural choices, and supporting robust engineering handoffs from experimentation to deployment.

## System Architecture
The core workflow is implemented under `Data Analysis for Hospitals/task/` as modular stages:

- **Data Pipeline**  
  Handles dataset loading, merging, and manifest generation to establish consistent inputs across experiments.
- **Preprocessing and Feature Engineering**  
  Applies data cleaning and constructs derived tabular features used by downstream models.
- **Model Training and Evaluation**  
  Trains predictive models and computes evaluation metrics for risk/outcome tasks.
- **Anomaly Detection and Early Warning**  
  Builds anomaly scores, simulates alerts over time, and measures detection latency.
- **Hardware-Aware Evaluation**  
  Benchmarks repeated runs, latency–accuracy trade-offs, and constrained early-warning scenarios.
- **Inference and Deployment Utilities**  
  Supports CPU inference benchmarking, ONNX export, streaming inference, and monitoring summaries.

## Repository Structure
- `Data Analysis for Hospitals/task/`  
  Main pipeline codebase, including CLI entrypoint and modular components for ingestion, modeling, evaluation, and deployment.
- `Data Analysis for Hospitals/test/`  
  CSV datasets used as local input data for the pipeline.
- `docs/`  
  Operational notes describing design motivations, trade-offs, assumptions, and limitations.
- `requirements.txt`  
  Python dependencies required to run the project.
- `LICENSE`  
  Project license file.

## Features
- Deterministic execution controls via global seeding and reproducibility context.
- CLI-driven experiment orchestration (`run`, `manifest`, and `early-warning-experiment`).
- Hardware-aware benchmarking for latency, memory/computation constraints, and deployment diagnostics.
- Early-warning scenario experimentation across configurable resource profiles.
- Artifact generation for manifests, logs, hardware profiles, and experiment summaries.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run commands from the repository root:

```bash
cd "Data Analysis for Hospitals/task"
python cli.py manifest
python cli.py run
python cli.py early-warning-experiment
```

Artifacts are written to `Data Analysis for Hospitals/task/artifacts/`.

## Reproducibility
Reproducible experimentation is supported through:

- **Configuration-driven execution** in `Data Analysis for Hospitals/task/config.py` (seed, dataset paths, feature columns, benchmark parameters).
- **Deterministic seeds** set at pipeline start to reduce run-to-run variance.
- **Persistent artifacts** (for example, dataset manifests, hardware profiles, and experiment logs) stored under `artifacts/` for auditability and comparison.

For stable comparisons, keep dependencies fixed (`requirements.txt`), use the same input data, and execute commands from a consistent environment.

## Related Projects
This repository is part of a broader engineering portfolio focused on:

- hardware-aware machine learning
- edge AI optimization
- deterministic ML pipelines
- production ML systems

Related repositories:

- `neural-network-from-scratch`
- `classification-of-handwritten-digits1`
- `edge-ai-hardware-optimization`
- `data-analysis-for-hospitals`
- `nba-data-preprocessing`
- `Data-Science-AI-Portfolio`

## Future Work
Potential extensions include:

- deployment and profiling on embedded/edge devices
- additional compression and optimization methods for inference efficiency
- expanded benchmarking frameworks with richer system-level telemetry
- broader validation across heterogeneous hardware environments

## License
This project is licensed under the terms specified in the `LICENSE` file.
