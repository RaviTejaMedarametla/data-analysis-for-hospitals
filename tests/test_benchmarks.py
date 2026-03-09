from pathlib import Path

from benchmarks.repeated_benchmark import run_repeated_benchmark


def test_repeated_benchmark_schema(tmp_path: Path):
    out = run_repeated_benchmark(lambda: sum(range(10)), tmp_path / "bench.json", runs=3, warmup_runs=1)
    assert {"mean_ms", "std_ms", "p95_ms"}.issubset(out.keys())
