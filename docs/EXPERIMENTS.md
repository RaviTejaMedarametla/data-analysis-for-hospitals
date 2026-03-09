# Experiments

## Benchmarks
- Repeated latency benchmark (`benchmarks/repeated_benchmark.py`)
- Memory profile (`benchmarks/memory_profile.py`)
- ONNX vs native CPU compare (`deployment/benchmark_cpu.py`)

## Ablations
Ablations vary feature groups (`all_features`, `no_body_metrics`, `vitals_only`) and export CSV tables to `Data Analysis for Hospitals/task/results/ablations.csv`.

## Interpretation
Use these artifacts as measured evidence; avoid unsupported claims about optimization when ONNX runtime is unavailable.
