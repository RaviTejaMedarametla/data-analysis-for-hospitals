from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression as _SkLogReg
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
    from sklearn.model_selection import RepeatedStratifiedKFold

    SKLEARN_AVAILABLE = True
except Exception:
    _SkLogReg = None
    SKLEARN_AVAILABLE = False


class SimpleLogisticModel:
    def __init__(self, lr: float = 0.01, epochs: int = 600):
        self.lr = lr
        self.epochs = epochs
        self.weights: np.ndarray | None = None
        self.feature_columns: list[str] | None = None

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -20, 20)))

    def fit(self, X: pd.DataFrame, y: pd.Series):
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


def _accuracy(y_true, y_pred):
    return float((np.array(y_true) == np.array(y_pred)).mean())


def _f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / max(tp + fp, 1.0)
    r = tp / max(tp + fn, 1.0)
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def _auc(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score))
    pos = y_true == 1
    n_pos = pos.sum()
    n_neg = len(y_true) - n_pos
    rank_sum = ranks[pos].sum()
    return float((rank_sum - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))


def _brier(y_true, y_prob):
    y_true = np.array(y_true, dtype=float)
    y_prob = np.array(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def _calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, 1, n_bins + 1)
    pred_points = []
    obs_points = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.any():
            pred_points.append(float(y_prob[mask].mean()))
            obs_points.append(float(y_true[mask].mean()))
    return np.array(obs_points), np.array(pred_points)


@dataclass
class ModelArtifacts:
    risk_model: object
    outcome_model: object
    X_test: pd.DataFrame
    y_risk_test: pd.Series
    y_outcome_test: pd.Series


def _prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df[feature_cols + ["hospital", "gender"]].copy()
    X = pd.get_dummies(X, columns=["hospital", "gender"], drop_first=False).astype(float)
    num = X[feature_cols]
    X[feature_cols] = (num - num.mean()) / (num.std(ddof=0) + 1e-6)
    return X


def _build_targets(df: pd.DataFrame, risk_target: str, outcome_target: str) -> tuple[pd.Series, pd.Series]:
    y_risk = df[risk_target].isin(["appendicitis", "pregnancy"]).astype(int)
    y_outcome = (df[outcome_target] == "t").astype(int)
    return y_risk, y_outcome


def _new_model(seed: int):
    if SKLEARN_AVAILABLE:
        return _SkLogReg(max_iter=2000, solver="lbfgs", random_state=seed)
    return SimpleLogisticModel()


def train_predictive_models(df: pd.DataFrame, feature_cols: list[str], risk_target: str, outcome_target: str) -> ModelArtifacts:
    X = _prepare_X(df, feature_cols)
    y_risk, y_outcome = _build_targets(df, risk_target, outcome_target)

    rng = np.random.default_rng(42)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * 0.75)
    tr, te = idx[:split], idx[split:]

    X_train, X_test = X.iloc[tr], X.iloc[te]
    y_risk_train, y_risk_test = y_risk.iloc[tr], y_risk.iloc[te]
    y_outcome_train, y_outcome_test = y_outcome.iloc[tr], y_outcome.iloc[te]

    risk_model = _new_model(42).fit(X_train, y_risk_train)
    outcome_model = _new_model(42).fit(X_train, y_outcome_train)
    return ModelArtifacts(risk_model, outcome_model, X_test, y_risk_test, y_outcome_test)


def evaluate_predictive_models(artifacts: ModelArtifacts) -> dict[str, float]:
    risk_pred = artifacts.risk_model.predict(artifacts.X_test)
    risk_prob = artifacts.risk_model.predict_proba(artifacts.X_test)[:, 1]
    outcome_pred = artifacts.outcome_model.predict(artifacts.X_test)
    outcome_prob = artifacts.outcome_model.predict_proba(artifacts.X_test)[:, 1]

    y_risk = artifacts.y_risk_test.values
    y_outcome = artifacts.y_outcome_test.values

    if SKLEARN_AVAILABLE:
        risk_acc = float(accuracy_score(y_risk, risk_pred))
        risk_f1 = float(f1_score(y_risk, risk_pred, zero_division=0))
        risk_auc = float(roc_auc_score(y_risk, risk_prob)) if len(np.unique(y_risk)) > 1 else 0.5
        risk_brier = float(brier_score_loss(y_risk, risk_prob))
        out_acc = float(accuracy_score(y_outcome, outcome_pred))
        out_f1 = float(f1_score(y_outcome, outcome_pred, zero_division=0))
        out_auc = float(roc_auc_score(y_outcome, outcome_prob)) if len(np.unique(y_outcome)) > 1 else 0.5
        out_brier = float(brier_score_loss(y_outcome, outcome_prob))
    else:
        risk_acc, risk_f1, risk_auc, risk_brier = _accuracy(y_risk, risk_pred), _f1(y_risk, risk_pred), _auc(y_risk, risk_prob), _brier(y_risk, risk_prob)
        out_acc, out_f1, out_auc, out_brier = _accuracy(y_outcome, outcome_pred), _f1(y_outcome, outcome_pred), _auc(y_outcome, outcome_prob), _brier(y_outcome, outcome_prob)

    return {
        "risk_accuracy": risk_acc,
        "risk_f1": risk_f1,
        "risk_auc": risk_auc,
        "risk_brier": risk_brier,
        "outcome_accuracy": out_acc,
        "outcome_f1": out_f1,
        "outcome_auc": out_auc,
        "outcome_brier": out_brier,
        "sample_count": float(len(artifacts.X_test)),
        "sklearn_available": float(SKLEARN_AVAILABLE),
    }


def repeated_stratified_cv_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    risk_target: str,
    seeds: list[int],
    n_splits: int,
    n_repeats: int,
    calibration_bins: int,
    output_path: Path,
) -> dict:
    X = _prepare_X(df, feature_cols)
    y_risk, _ = _build_targets(df, risk_target=risk_target, outcome_target="blood_test")

    fold_rows: list[dict[str, float]] = []
    all_prob = []
    all_true = []

    for seed in seeds:
        if SKLEARN_AVAILABLE:
            splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
            split_iter = splitter.split(X, y_risk)
        else:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(X))
            split_iter = []
            for _ in range(n_splits * n_repeats):
                rng.shuffle(idx)
                cut = int(0.8 * len(idx))
                split_iter.append((idx[:cut], idx[cut:]))

        for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
            model = _new_model(seed).fit(X.iloc[train_idx], y_risk.iloc[train_idx])
            prob = model.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            y_fold = y_risk.iloc[test_idx].values
            row = {
                "seed": float(seed),
                "fold": float(fold_id),
                "accuracy": _accuracy(y_fold, pred),
                "f1": _f1(y_fold, pred),
                "auc": _auc(y_fold, prob),
                "brier": _brier(y_fold, prob),
            }
            fold_rows.append(row)
            all_prob.extend(prob.tolist())
            all_true.extend(y_fold.tolist())

    fold_df = pd.DataFrame(fold_rows)
    prob_true, prob_pred = _calibration_curve(np.array(all_true), np.array(all_prob), n_bins=calibration_bins)
    reliability = pd.DataFrame({"predicted_prob_bin": prob_pred, "observed_rate": prob_true})

    report = {
        "metrics_distribution": {
            metric: {
                "mean": float(fold_df[metric].mean()),
                "std": float(fold_df[metric].std(ddof=1) if len(fold_df) > 1 else 0.0),
                "min": float(fold_df[metric].min()),
                "max": float(fold_df[metric].max()),
            }
            for metric in ["accuracy", "f1", "auc", "brier"]
        },
        "fold_count": int(len(fold_df)),
        "sklearn_available": float(SKLEARN_AVAILABLE),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(output_path.parent / "predictive_cv_folds.csv", index=False)
    reliability.to_csv(output_path.parent / "predictive_reliability_curve.csv", index=False)
    (output_path.parent / "predictive_cv_report.json").write_text(json.dumps(report, indent=2))

    return {
        "summary": report,
        "fold_artifact": str(output_path.parent / "predictive_cv_folds.csv"),
        "reliability_artifact": str(output_path.parent / "predictive_reliability_curve.csv"),
        "report_artifact": str(output_path.parent / "predictive_cv_report.json"),
    }
