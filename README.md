# Data Analysis for Hospitals

Comprehensive overhaul of a reproducible hospital analytics pipeline with explicit benchmarking evidence.

## What changed
- Refactored monolithic CLI flow into modular pipeline stages (`pipeline/train.py`, `pipeline/evaluate.py`, `pipeline/anomaly.py`, `pipeline/deploy.py`, `pipeline/run.py`).
- Added measured deployment telemetry with ONNX Runtime (`deployment/onnx_inference.py`) and CPU comparison script (`deployment/benchmark_cpu.py`).
- Added benchmark suite under `benchmarks/` for repeated latency, memory profiling, stage latency breakdown, and model comparison.
- Added baseline models (`modeling/baselines.py`) and ablation studies (`experiments/ablations.py`).
- Added reproducibility assets: configurable split/benchmark seeds, lock file, and full reproduction script.
- Expanded tests and CI to run full pytest coverage.

## Run
```bash
cd "Data Analysis for Hospitals/task"
python cli.py manifest
python cli.py run
python cli.py ablation
```

## Reproduce everything
```bash
bash reproduce_all.sh
```

## Documentation
- `docs/RESEARCH_TASK.md`
- `docs/DATASET.md`
- `docs/EXPERIMENTS.md`
