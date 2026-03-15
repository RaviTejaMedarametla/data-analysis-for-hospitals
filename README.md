# Data Analysis for Hospitals

A research‑oriented framework for analyzing electronic health records (EHR), predicting patient outcomes (readmission, length‑of‑stay), detecting anomalies in real‑time, and deploying models under hardware constraints. The project emphasises reproducibility, hardware‑aware evaluation, and end‑to‑end pipeline integration.

## Features

- **Data ingestion** – load and version‑control hospital datasets (general, prenatal, sports).
- **Preprocessing & cleaning** – handle missing values, standardize columns, and engineer features (age, BMI, risk bands).
- **Predictive modeling** – from‑scratch logistic regression for risk and outcome prediction, with custom evaluation metrics.
- **Anomaly detection** – outlier detection using Z‑scores and early‑warning simulation.
- **Hardware‑aware evaluation** – measure latency, memory usage, and energy proxy; enforce constraints.
- **Deployment simulation** – export models to ONNX, benchmark CPU vs. ONNXRuntime.
- **Streaming & real‑time inference** – process data in chunks with adaptive memory management.
- **Reproducibility** – deterministic seeds, configuration dataclasses, and comprehensive logging.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:
```bash
cd task
python cli.py run
```

Create a dataset manifest:
```bash
python cli.py manifest
```

Run ablation studies:
```bash
python cli.py ablation
```

Reproduce all experiments (from root):
```bash
bash reproduce_all.sh
```

## Project Structure

- `task/` – main package containing all modules.
  - `anomaly_detection/` – outlier detection and early warning.
  - `benchmarks/` – latency, memory, and repeated benchmarks.
  - `deployment/` – ONNX export, CPU inference, monitoring.
  - `evaluation/` – metrics, statistics, hardware profiling.
  - `experiments/` – ablation studies.
  - `feature_engineering/` – feature creation from raw data.
  - `ingestion/` – data loading and versioning.
  - `modeling/` – predictive models (risk, outcome, baselines).
  - `pipeline/` – orchestration of training, evaluation, deployment.
  - `preprocessing/` – data cleaning.
  - `real_time/` – streaming inference.
  - `utils/` – energy estimation, hardware monitoring, reproducibility.
- `tests/` – unit and integration tests.
- `pytest.ini` – pytest configuration.
- `reproduce_all.sh` – shell script to run all experiments and tests.
- `requirements.txt` / `requirements-lock.txt` – pinned dependencies.

## License

This project is released under the MIT License. See `LICENSE` for details.
