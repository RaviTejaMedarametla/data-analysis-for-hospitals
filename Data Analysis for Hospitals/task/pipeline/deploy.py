from __future__ import annotations

from config import CONFIG
from deployment.benchmark_cpu import benchmark_cpu
from deployment.cpu_inference import run_cpu_inference
from deployment.onnx_export import export_pipeline_to_onnx


def run_deployment_pipeline(artifacts):
    inference_stats = run_cpu_inference(artifacts.risk_model, artifacts.X_test)
    onnx_info = export_pipeline_to_onnx(
        artifacts.risk_model,
        CONFIG.output_dir / "risk_model.onnx",
        n_features=len(artifacts.X_test.columns),
    )
    benchmark = None
    if onnx_info["success"]:
        benchmark = benchmark_cpu(
            artifacts.risk_model,
            artifacts.X_test,
            CONFIG.output_dir / "risk_model.onnx",
            CONFIG.results_dir / "cpu_benchmark.json",
        )
    return {"cpu_inference": inference_stats, "onnx_export": onnx_info, "cpu_benchmark": benchmark}
