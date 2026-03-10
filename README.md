# Data Analysis for Hospitals

A reproducible hospital analytics project that trains risk/outcome models, detects anomalies, and benchmarks deployment performance.

## Project overview
- Ingests and harmonizes multi-source hospital datasets (general, prenatal, sports) into a unified analysis-ready table.
- Trains predictive models for risk and outcome tasks, then evaluates accuracy, F1, and AUC.
- Runs anomaly and early-warning analysis to flag unusual operational patterns.
- Exports deployment artifacts and compares local CPU vs ONNX Runtime inference performance.
- Supports repeatable experiments through fixed seeds, deterministic settings, and documented reproduction steps.

## Repository structure
- `Data Analysis for Hospitals/task/cli.py`: entrypoint for pipeline commands.
- `Data Analysis for Hospitals/task/pipeline/`: orchestrated train/evaluate/anomaly/deploy workflow stages.
- `Data Analysis for Hospitals/task/modeling/`: predictive models and baselines.
- `Data Analysis for Hospitals/task/benchmarks/`: latency and memory benchmarking utilities.
- `Data Analysis for Hospitals/task/deployment/`: ONNX export/inference and deployment benchmarking.
- `docs/`: dataset, experiments, and research notes.

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
