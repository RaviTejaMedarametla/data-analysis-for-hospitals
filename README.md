# Hardware-Aware Early Warning and Predictive Analytics for Hospital Data

This repository contains a production-oriented analytics pipeline for tabular hospital datasets.
It is designed to execute end-to-end on CPU-constrained environments while preserving deterministic behavior and auditability.

## Scope

The project focuses on four operational goals:

- robust data ingestion and schema normalization,
- predictive risk modeling and anomaly-triggered early warning,
- streaming inference under constrained latency budgets,
- deployment diagnostics for resource and reliability monitoring.

The implementation emphasizes incremental validation and reproducible outputs over one-off model results.

## Repository layout

Core code is located in `Data Analysis for Hospitals/task/`.

- `ingestion/`: input loading and dataset manifest generation.
- `preprocessing/`: cleaning and consistency checks.
- `feature_engineering/`: derived feature construction.
- `modeling/`: predictive and risk-band modeling.
- `anomaly_detection/`: outlier detection and early-warning simulation.
- `real_time/`: streaming utilities and online scoring.
- `deployment/`: CPU inference, ONNX export, and monitoring summaries.
- `evaluation/`: statistical metrics, benchmark utilities, and experiment summarization.
- `utils/`: reproducibility, logging, and hardware/energy estimation helpers.

## Pipeline commands

```bash
cd "Data Analysis for Hospitals/task"
python cli.py manifest
python cli.py run
python cli.py early-warning-experiment
```

Generated artifacts are written to `Data Analysis for Hospitals/task/artifacts/`.

## Design motivations and trade-offs

- **CPU-first execution:** prioritizes compatibility with common deployment targets; GPU-only optimizations are intentionally out of scope.
- **Explicit hardware constraints:** memory limits and compute budgets are treated as first-class experiment parameters.
- **Model simplicity vs. latency:** simpler models reduce inference cost but may underfit rare patterns; benchmark outputs expose this trade-off.
- **Streaming vs. batch behavior:** stream chunking improves online responsiveness but can increase overhead and jitter at very small chunk sizes.

## Performance constraints

- Throughput and latency depend on stream chunk size, feature dimensionality, and configured compute budget.
- Memory pressure is sensitive to effective batch size and precision assumptions.
- Repeated benchmark runs are required because single-run latency is noisy on shared or throttled hosts.

## Failure modes and bottlenecks

Typical operational risks:

- schema drift or missing columns in input CSV files,
- high false-positive rates in anomaly-triggered alerts,
- latency spikes from oversized stream chunks,
- degraded detection quality under strict memory/compute envelopes,
- serialization/export mismatch during ONNX conversion.

## Assumptions

- Input CSV files follow the expected column schema used in feature generation.
- Runtime has sufficient permissions to create files in the configured artifact directory.
- The environment provides compatible versions of NumPy, pandas, and scikit-learn dependencies.

## Limitations

- Energy and hardware functions provide coarse estimates, not device-calibrated measurements.
- Default benchmarks are short and should be expanded for production sign-off.
- Synthetic event construction in latency evaluation is a proxy and not a substitute for externally labeled event timelines.


## Additional documentation

- Operational constraints and deployment considerations: `docs/OPERATIONS.md`.


## Dependency and CI policy

- The repository uses standard Python tooling (`pytest`, `unittest`) and has no platform-specific testing dependencies.
- CI targets Python 3.10 with pinned dependencies for reproducible execution.
- Runtime and tests are designed for deterministic behavior via explicit seed and threading environment controls.
