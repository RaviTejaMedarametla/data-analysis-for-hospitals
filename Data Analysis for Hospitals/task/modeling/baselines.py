from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@dataclass
class BaselineResult:
    model_name: str
    accuracy: float
    f1: float
    auc: float


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    seed: int = 42,
) -> list[BaselineResult]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed),
    }
    results: list[BaselineResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        score = model.predict_proba(X_test)[:, 1]
        results.append(
            BaselineResult(
                model_name=name,
                accuracy=float(accuracy_score(y_test, pred)),
                f1=float(f1_score(y_test, pred, zero_division=0)),
                auc=float(roc_auc_score(y_test, score)) if y_test.nunique() > 1 else 0.5,
            )
        )
    return results
