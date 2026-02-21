from __future__ import annotations

import pandas as pd


AGE_BINS = [0, 15, 35, 55, 70, 80]
AGE_LABELS = ["0-15", "15-35", "35-55", "55-70", "70-80"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["age_range"] = pd.cut(feat["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    feat["is_adult"] = (feat["age"] >= 18).astype(int)
    feat["bmi_risk"] = pd.cut(feat["bmi"], bins=[-1, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    feat["bmi_risk"] = feat["bmi_risk"].fillna(0)
    return feat
