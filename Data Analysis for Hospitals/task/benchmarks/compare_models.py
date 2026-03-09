from __future__ import annotations

import pandas as pd

from modeling.baselines import train_and_evaluate_baselines


def compare_models(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    baseline_results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test)
    return pd.DataFrame([r.__dict__ for r in baseline_results])
