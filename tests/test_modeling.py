from pipeline.train import run_training_pipeline
from modeling.predictive import evaluate_predictive_models


def test_training_and_prediction_shapes():
    _, artifacts = run_training_pipeline()
    metrics = evaluate_predictive_models(artifacts)
    assert "risk_accuracy" in metrics
    assert len(artifacts.X_test) > 0
