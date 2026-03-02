# Operations and Deployment Notes

## System design motivations

The pipeline is organized as explicit stages to isolate failure domains: ingestion, preprocessing, feature generation, model evaluation, and deployment diagnostics. This keeps data-quality errors from being conflated with model-quality regressions.

## Architectural trade-offs

- Stage boundaries add I/O and serialization overhead, but improve debuggability and restartability.
- CPU-first inference reduces hardware variance, but limits peak throughput.
- Conservative default batch sizes improve stability under memory pressure at the cost of total runtime.

## Assumptions

- Input datasets are trusted CSV files with expected column names.
- Artifact storage is local filesystem-based and single-writer.
- Benchmarking is performed on relatively stable host resources.

## Limitations

- Throughput estimates are approximate and should be validated with host-level profilers.
- Energy and quantization metrics are model-based estimates and not physical measurements.
- Monitoring summaries are operational indicators, not incident response automation.

## Failure modes and bottlenecks

- Missing or malformed input columns will fail downstream feature operations.
- Alert threshold drift can inflate false positive rates.
- Long-running benchmark loops are sensitive to shared CPU contention.
- ONNX export can fail for unsupported estimator shapes or operators.
