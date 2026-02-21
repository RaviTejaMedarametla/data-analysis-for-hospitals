from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemConfig:
    random_seed: int = 42
    test_size: float = 0.25
    data_dir: Path = Path(__file__).resolve().parent.parent / "test"
    output_dir: Path = Path(__file__).resolve().parent / "artifacts"
    stream_chunk_size: int = 16
    stream_interval_ms: int = 10
    hardware_memory_limit_mb: int = 256
    hardware_compute_budget: int = 10_000
    benchmark_runs: int = 5
    confidence_level: float = 0.95
    feature_columns: list[str] = field(
        default_factory=lambda: ["age", "height", "weight", "bmi", "children", "months"]
    )
    target_risk: str = "diagnosis"
    target_outcome: str = "blood_test"
    experiment_memory_limits_mb: list[int] = field(default_factory=lambda: [64, 128, 256])
    experiment_compute_budgets: list[int] = field(default_factory=lambda: [2_000, 5_000, 10_000])
    experiment_stream_speeds_ms: list[int] = field(default_factory=lambda: [5, 10, 20])


CONFIG = SystemConfig()
CONFIG.output_dir.mkdir(parents=True, exist_ok=True)
