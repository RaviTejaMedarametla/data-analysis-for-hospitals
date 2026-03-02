from __future__ import annotations

import argparse
import itertools
import json
import pandas as pd

from config import CONFIG
from ingestion.loader import load_hospital_data, merge_hospital_data
from ingestion.versioning import create_dataset_manifest
from preprocessing.cleaning import clean_hospital_data
from feature_engineering.features import build_features
from modeling.predictive import train_predictive_models, evaluate_predictive_models
from anomaly_detection.detectors import OutlierDetector
from anomaly_detection.early_warning import simulate_early_warning, evaluate_detection_latency
from real_time.streaming import compare_batch_vs_streaming
from deployment.cpu_inference import run_cpu_inference
from deployment.onnx_export import export_pipeline_to_onnx
from deployment.monitoring import build_monitoring_summary
from evaluation.benchmark import run_repeated_benchmark, benchmark_table_metrics
from evaluation.metrics import latency_accuracy_tradeoff
from evaluation.hardware_profile import build_hardware_profile_table, write_hardware_profile_artifacts
from evaluation.early_warning_experiment import (
    ConstraintScenario,
    run_hardware_constrained_early_warning_experiment,
    summarize_experiment,
)
from utils.reproducibility import set_global_seed, reproducibility_context
from utils.logging_utils import log_experiment
from utils.hardware import HardwareProfile, auto_adjust_batch_size, compute_utilization
from utils.energy import compare_precision_energy
from modeling.risk import stratify_risk, summarize_risk_bands
from real_time.inference import run_streaming_inference


def _build_scenarios() -> list[ConstraintScenario]:
    return [
        ConstraintScenario(memory_limit_mb=m, compute_budget=c, stream_interval_ms=s)
        for m, c, s in itertools.product(
            CONFIG.experiment_memory_limits_mb,
            CONFIG.experiment_compute_budgets,
            CONFIG.experiment_stream_speeds_ms,
        )
    ]


def run_pipeline() -> dict:
    set_global_seed(CONFIG.random_seed)

    datasets = load_hospital_data(CONFIG.data_dir)
    merged = merge_hospital_data(datasets)
    clean = clean_hospital_data(merged)
    feat = build_features(clean)

    artifacts = train_predictive_models(feat, CONFIG.feature_columns, CONFIG.target_risk, CONFIG.target_outcome)
    model_metrics = evaluate_predictive_models(artifacts)

    detector = OutlierDetector(random_state=CONFIG.random_seed).fit(feat[CONFIG.feature_columns])
    anomaly = detector.detect(feat[CONFIG.feature_columns])

    timestamps = pd.date_range("2025-01-01", periods=len(feat), freq="min")
    early_warning = simulate_early_warning(anomaly["anomaly_score"], timestamps, anomaly["anomaly_score"].quantile(0.9))
    synthetic_events = (anomaly["anomaly_score"] > anomaly["anomaly_score"].quantile(0.95)).astype(int)
    latency = evaluate_detection_latency(anomaly["anomaly_score"], synthetic_events, timestamps)

    stream_stats = compare_batch_vs_streaming(feat[CONFIG.feature_columns], lambda x: x.assign(score=x.sum(axis=1)), CONFIG.stream_chunk_size)

    hardware = HardwareProfile(CONFIG.hardware_memory_limit_mb, CONFIG.hardware_compute_budget)
    adjusted_batch = auto_adjust_batch_size(128, len(CONFIG.feature_columns), hardware)
    utilization = compute_utilization(len(feat) * len(CONFIG.feature_columns), hardware)

    inference_stats = run_cpu_inference(artifacts.risk_model, artifacts.X_test)
    onnx_ok = export_pipeline_to_onnx(artifacts.risk_model, CONFIG.output_dir / "risk_model.onnx", n_features=len(CONFIG.feature_columns))

    risk_probabilities = pd.Series(artifacts.risk_model.predict_proba(artifacts.X_test)[:, 1], index=artifacts.X_test.index)
    risk_frame = stratify_risk(risk_probabilities)
    streaming_output = run_streaming_inference(artifacts.X_test, artifacts.risk_model, CONFIG.stream_chunk_size)
    monitoring = build_monitoring_summary(
        alert_flags=(risk_frame["risk_band"] == "high").astype(int),
        risk_probabilities=risk_frame["risk_probability"],
        stream_latency_ms_per_row=stream_stats["stream_latency_ms_per_row"],
    )

    bench = run_repeated_benchmark(
        lambda: evaluate_predictive_models(artifacts),
        metric_key="risk_accuracy",
        runs=CONFIG.benchmark_runs,
        confidence=CONFIG.confidence_level,
    )
    tradeoff = latency_accuracy_tradeoff(model_metrics["risk_accuracy"], inference_stats["inference_latency_ms"])
    energy = compare_precision_energy(runtime_s=inference_stats["inference_latency_ms"] / 1000, batch_size=adjusted_batch)

    hardware_profile = build_hardware_profile_table(
        feature_count=len(CONFIG.feature_columns),
        batch_size=adjusted_batch,
        stream_interval_ms=CONFIG.stream_interval_ms,
    )
    hardware_profile_artifacts = write_hardware_profile_artifacts(hardware_profile, CONFIG.output_dir)

    scenarios = _build_scenarios()
    exp_df, exp_artifacts = run_hardware_constrained_early_warning_experiment(
        df=feat,
        feature_cols=CONFIG.feature_columns,
        target_col=CONFIG.target_risk,
        scenarios=scenarios,
        output_dir=CONFIG.output_dir,
    )
    exp_summary = summarize_experiment(exp_df)
    exp_benchmark = benchmark_table_metrics(
        exp_df,
        ["detection_latency_s", "prediction_accuracy", "false_positive_rate", "detection_quality"],
        confidence=CONFIG.confidence_level,
    )

    manifest = create_dataset_manifest(CONFIG.data_dir, CONFIG.output_dir / "dataset_manifest.json")

    results = {
        "reproducibility": reproducibility_context(CONFIG),
        "predictive_metrics": model_metrics,
        "anomaly_alerts": early_warning,
        "detection_latency_s": latency,
        "streaming": stream_stats,
        "hardware": {"adjusted_batch_size": adjusted_batch, "compute_utilization": utilization},
        "cpu_inference": inference_stats,
        "onnx_exported": onnx_ok,
        "benchmark": bench.__dict__,
        "latency_accuracy_tradeoff": tradeoff,
        "energy": energy,
        "hardware_profile": hardware_profile,
        "hardware_profile_artifacts": hardware_profile_artifacts,
        "risk_modeling": {
            "risk_band_summary": summarize_risk_bands(risk_frame),
            "high_risk_count": int((risk_frame["risk_band"] == "high").sum()),
        },
        "streaming_inference": {
            "records_scored": int(len(streaming_output)),
            "high_risk_predictions": int(streaming_output["risk_label"].sum()),
        },
        "deployment_monitoring": monitoring,
        "early_warning_hardware_experiment": {
            "scenario_count": len(exp_df),
            "summary": exp_summary,
            "benchmark": exp_benchmark,
            "artifacts": exp_artifacts,
        },
        "dataset_manifest_files": len(manifest["files"]),
    }
    log_experiment(results, CONFIG.output_dir / "experiment_log.json")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Healthcare AI research pipeline")
    parser.add_argument("command", choices=["run", "manifest", "early-warning-experiment"], help="Pipeline command")
    args = parser.parse_args()

    if args.command == "manifest":
        manifest = create_dataset_manifest(CONFIG.data_dir, CONFIG.output_dir / "dataset_manifest.json")
        print(json.dumps(manifest, indent=2))
        return

    if args.command == "early-warning-experiment":
        datasets = load_hospital_data(CONFIG.data_dir)
        merged = merge_hospital_data(datasets)
        clean = clean_hospital_data(merged)
        feat = build_features(clean)
        exp_df, exp_artifacts = run_hardware_constrained_early_warning_experiment(
            df=feat,
            feature_cols=CONFIG.feature_columns,
            target_col=CONFIG.target_risk,
            scenarios=_build_scenarios(),
            output_dir=CONFIG.output_dir,
        )
        print(json.dumps({"summary": summarize_experiment(exp_df), "benchmark": benchmark_table_metrics(exp_df, ["detection_latency_s", "prediction_accuracy", "false_positive_rate", "detection_quality"], confidence=CONFIG.confidence_level), "artifacts": exp_artifacts}, indent=2, default=float))
        return

    print(json.dumps(run_pipeline(), indent=2, default=float))


if __name__ == "__main__":
    main()
