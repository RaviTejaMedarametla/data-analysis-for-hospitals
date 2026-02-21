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
from modeling.predictive import (
    train_predictive_models,
    evaluate_predictive_models,
    repeated_stratified_cv_report,
)
from anomaly_detection.detectors import OutlierDetector, compare_detectors
from anomaly_detection.early_warning import simulate_early_warning, evaluate_detection_latency
from real_time.streaming import compare_batch_vs_streaming
from deployment.cpu_inference import run_cpu_inference
from deployment.onnx_export import export_pipeline_to_onnx, onnx_parity_check
from evaluation.benchmark import run_repeated_benchmark, benchmark_table_metrics
from evaluation.metrics import latency_accuracy_tradeoff
from evaluation.early_warning_experiment import (
    ConstraintScenario,
    run_hardware_constrained_early_warning_experiment,
    summarize_experiment,
)
from evaluation.time_aware import (
    rolling_origin_splits,
    detection_delay_distribution,
    false_alarms_per_hour,
    precision_recall_at_budget,
)
from utils.reproducibility import set_global_seed
from utils.logging_utils import log_experiment
from utils.hardware import HardwareProfile, auto_adjust_batch_size, compute_utilization
from utils.energy import compare_precision_energy
from utils.telemetry import measure_block_energy_runtime
from utils.runtime_metadata import write_runtime_metadata


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
    cv_report = repeated_stratified_cv_report(
        df=feat,
        feature_cols=CONFIG.feature_columns,
        risk_target=CONFIG.target_risk,
        seeds=CONFIG.cv_seeds,
        n_splits=CONFIG.cv_splits,
        n_repeats=CONFIG.cv_repeats,
        calibration_bins=CONFIG.calibration_bins,
        output_path=CONFIG.output_dir / "predictive_cv_report.json",
    )

    detector = OutlierDetector(random_state=CONFIG.random_seed).fit(feat[CONFIG.feature_columns])
    anomaly = detector.detect(feat[CONFIG.feature_columns])

    timestamps = pd.date_range("2025-01-01", periods=len(feat), freq="min")
    early_warning = simulate_early_warning(anomaly["anomaly_score"], timestamps, anomaly["anomaly_score"].quantile(0.9))
    synthetic_events = (anomaly["anomaly_score"] > anomaly["anomaly_score"].quantile(0.95)).astype(int)
    latency = evaluate_detection_latency(anomaly["anomaly_score"], synthetic_events, timestamps)

    delay_dist = detection_delay_distribution(
        scores=anomaly["anomaly_score"],
        events=synthetic_events,
        timestamps=timestamps,
        threshold=float(anomaly["anomaly_score"].quantile(0.9)),
    )
    alert_series = (anomaly["anomaly_score"] >= anomaly["anomaly_score"].quantile(0.9)).astype(int)
    false_alert_rate_hr = false_alarms_per_hour(alert_series, synthetic_events, timestamps)
    pr_budget = precision_recall_at_budget(anomaly["anomaly_score"], synthetic_events, CONFIG.alert_budget_fractions)

    stream_stats = compare_batch_vs_streaming(
        feat[CONFIG.feature_columns], lambda x: x.assign(score=x.sum(axis=1)), CONFIG.stream_chunk_size
    )

    hardware = HardwareProfile(CONFIG.hardware_memory_limit_mb, CONFIG.hardware_compute_budget)
    adjusted_batch = auto_adjust_batch_size(128, len(CONFIG.feature_columns), hardware)
    utilization = compute_utilization(len(feat) * len(CONFIG.feature_columns), hardware)

    inference_stats, telemetry = measure_block_energy_runtime(lambda: run_cpu_inference(artifacts.risk_model, artifacts.X_test))
    onnx_ok = export_pipeline_to_onnx(
        artifacts.risk_model,
        CONFIG.output_dir / "risk_model.onnx",
        n_features=artifacts.X_test.shape[1],
    )
    onnx_parity = onnx_parity_check(
        model=artifacts.risk_model,
        onnx_path=CONFIG.output_dir / "risk_model.onnx",
        X=artifacts.X_test,
    )

    bench = run_repeated_benchmark(
        lambda: evaluate_predictive_models(train_predictive_models(feat, CONFIG.feature_columns, CONFIG.target_risk, CONFIG.target_outcome)),
        metric_key="risk_accuracy",
        runs=CONFIG.benchmark_runs,
    )
    tradeoff = latency_accuracy_tradeoff(model_metrics["risk_accuracy"], inference_stats["inference_latency_ms"])
    energy = compare_precision_energy(runtime_s=inference_stats["inference_latency_ms"] / 1000, batch_size=adjusted_batch)
    estimated_vs_measured = {
        "estimated_fp32_j": energy["fp32_joules"],
        "measured_j": telemetry["measured_energy_joules"],
        "absolute_error_j": abs(energy["fp32_joules"] - telemetry["measured_energy_joules"])
        if str(telemetry["measured_energy_joules"]) != "nan"
        else float("nan"),
    }

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
        exp_df, ["detection_latency_s", "prediction_accuracy", "false_positive_rate", "detection_quality"]
    )

    y_true_events = feat[CONFIG.target_risk].isin(["appendicitis", "pregnancy"]).astype(int)
    detector_compare = compare_detectors(feat[CONFIG.feature_columns], y_true_events)
    detector_compare_path = CONFIG.output_dir / "anomaly_detector_comparison.csv"
    detector_compare.to_csv(detector_compare_path, index=False)

    temporal_splits = rolling_origin_splits(
        len(feat),
        min_train=CONFIG.temporal_min_train,
        horizon=CONFIG.temporal_horizon,
        step=CONFIG.temporal_step,
    )

    manifest = create_dataset_manifest(CONFIG.data_dir, CONFIG.output_dir / "dataset_manifest.json")
    runtime_meta = write_runtime_metadata(CONFIG.output_dir / "runtime_metadata.json")

    results = {
        "predictive_metrics": model_metrics,
        "predictive_cv": cv_report,
        "anomaly_alerts": early_warning,
        "detection_latency_s": latency,
        "detection_delay_distribution": delay_dist,
        "false_alarms_per_hour": false_alert_rate_hr,
        "precision_recall_at_budget": pr_budget,
        "streaming": stream_stats,
        "hardware": {"adjusted_batch_size": adjusted_batch, "compute_utilization": utilization},
        "telemetry": telemetry,
        "cpu_inference": inference_stats,
        "estimated_vs_measured_energy": estimated_vs_measured,
        "onnx_exported": onnx_ok,
        "onnx_parity": onnx_parity,
        "benchmark": bench.__dict__,
        "latency_accuracy_tradeoff": tradeoff,
        "energy": energy,
        "early_warning_hardware_experiment": {
            "scenario_count": len(exp_df),
            "summary": exp_summary,
            "benchmark": exp_benchmark,
            "artifacts": exp_artifacts,
        },
        "anomaly_detector_comparison_artifact": str(detector_compare_path),
        "temporal_split_count": len(temporal_splits),
        "dataset_manifest_files": len(manifest["files"]),
        "runtime_metadata_artifact": runtime_meta,
        "deployment_narrative": {
            "icu_monitoring": "Latency-alert tradeoffs can map to ICU bedside alarm sensitivity settings.",
            "wearable_edge_ai": "Memory/compute sweeps approximate edge battery and processor constraints.",
            "medical_telemetry": "Async queue/drop metrics emulate telemetry ingestion under bursty network load.",
        },
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
        print(
            json.dumps(
                {
                    "summary": summarize_experiment(exp_df),
                    "benchmark": benchmark_table_metrics(
                        exp_df, ["detection_latency_s", "prediction_accuracy", "false_positive_rate", "detection_quality"]
                    ),
                    "artifacts": exp_artifacts,
                },
                indent=2,
                default=float,
            )
        )
        return

    print(json.dumps(run_pipeline(), indent=2, default=float))


if __name__ == "__main__":
    main()
