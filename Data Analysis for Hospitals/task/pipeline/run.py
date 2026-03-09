from __future__ import annotations

from config import CONFIG
from ingestion.versioning import create_dataset_manifest
from pipeline.anomaly import run_anomaly_pipeline
from pipeline.deploy import run_deployment_pipeline
from pipeline.evaluate import run_evaluation_pipeline
from pipeline.train import run_training_pipeline
from utils.logging_utils import log_experiment
from utils.reproducibility import reproducibility_context, set_global_seed


def run_pipeline() -> dict:
    set_global_seed(CONFIG.random_seed)
    feat, artifacts = run_training_pipeline()
    metrics, benchmark = run_evaluation_pipeline(artifacts)
    anomaly = run_anomaly_pipeline(feat)
    deployment = run_deployment_pipeline(artifacts)
    manifest = create_dataset_manifest(CONFIG.data_dir, CONFIG.output_dir / "dataset_manifest.json")
    results = {
        "reproducibility": reproducibility_context(CONFIG),
        "predictive_metrics": metrics,
        "benchmark": benchmark,
        "anomaly_alerts": anomaly["alerts"],
        "detection_latency_s": anomaly["latency"],
        **deployment,
        "dataset_manifest_files": len(manifest["files"]),
    }
    log_experiment(results, CONFIG.output_dir / "experiment_log.json")
    return results
