from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from utils.numpy_compat import integrate


class SimpleLogisticModel:
    def __init__(self, lr: float = 0.01, epochs: int = 600):
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.feature_columns: list[str] | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleLogisticModel":
        self.feature_columns = list(X.columns)
        x = X.values.astype(float)
        yv = y.values.astype(float)
        self.weights = np.zeros(x.shape[1] + 1)
        for _ in range(self.epochs):
            logits = x @ self.weights[1:] + self.weights[0]
            preds = self._sigmoid(logits)
            err = preds - yv
            self.weights[0] -= self.lr * err.mean()
            self.weights[1:] -= self.lr * (x.T @ err) / len(x)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x = X[self.feature_columns].values.astype(float)
        logits = x @ self.weights[1:] + self.weights[0]
        p1 = self._sigmoid(logits)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]

    n_pos = int((y_sorted == 1).sum())
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)

    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))

    return float(integrate(tpr, fpr))


@dataclass
class ModelArtifacts:
    risk_model: SimpleLogisticModel
    outcome_model: SimpleLogisticModel
    X_test: pd.DataFrame
    y_risk_test: pd.Series
    y_outcome_test: pd.Series


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols + ["hospital", "gender"]].copy()
    X = pd.get_dummies(X, columns=["hospital", "gender"], drop_first=False).astype(float)
    num = X[feature_cols]
    X[feature_cols] = (num - num.mean()) / (num.std(ddof=0) + 1e-6)
    return X


def train_predictive_models(df: pd.DataFrame, feature_cols: list[str], risk_target: str, outcome_target: str, split_seed: int = 42) -> ModelArtifacts:
    X = _prepare_X(df, feature_cols)
    y_risk = df[risk_target].isin(["appendicitis", "pregnancy"]).astype(int)
    y_outcome = (df[outcome_target] == "t").astype(int)

    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    split = int(len(X) * 0.75)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_risk_train = y_risk.iloc[train_idx]
    y_risk_test = y_risk.iloc[test_idx]
    y_outcome_train = y_outcome.iloc[train_idx]
    y_outcome_test = y_outcome.iloc[test_idx]

    risk_model = SimpleLogisticModel().fit(X_train, y_risk_train)
    outcome_model = SimpleLogisticModel().fit(X_train, y_outcome_train)

    return ModelArtifacts(risk_model, outcome_model, X_test, y_risk_test, y_outcome_test)


def evaluate_predictive_models(artifacts: ModelArtifacts) -> dict[str, float]:
    risk_pred = artifacts.risk_model.predict(artifacts.X_test)
    risk_prob = artifacts.risk_model.predict_proba(artifacts.X_test)[:, 1]
    outcome_pred = artifacts.outcome_model.predict(artifacts.X_test)
    outcome_prob = artifacts.outcome_model.predict_proba(artifacts.X_test)[:, 1]

    y_risk = artifacts.y_risk_test.values
    y_outcome = artifacts.y_outcome_test.values

    return {
        "risk_accuracy": _accuracy(y_risk, risk_pred),
        "risk_f1": _f1(y_risk, risk_pred),
        "risk_auc": _auc(y_risk, risk_prob),
        "outcome_accuracy": _accuracy(y_outcome, outcome_pred),
        "outcome_f1": _f1(y_outcome, outcome_pred),
        "outcome_auc": _auc(y_outcome, outcome_prob),
        "sample_count": float(len(artifacts.X_test)),
    }
