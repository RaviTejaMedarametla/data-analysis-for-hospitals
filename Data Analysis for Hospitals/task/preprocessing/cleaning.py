from __future__ import annotations

import pandas as pd


NUMERIC_FILL_COLUMNS = ["bmi", "children", "months"]
TEST_COLUMNS = ["blood_test", "ecg", "ultrasound", "mri", "xray"]


def clean_hospital_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean["gender"] = clean["gender"].replace({"male": "m", "female": "f", "man": "m", "woman": "f"})
    clean["gender"] = clean["gender"].fillna("f")
    clean[NUMERIC_FILL_COLUMNS] = clean[NUMERIC_FILL_COLUMNS].fillna(0)
    for col in TEST_COLUMNS:
        clean[col] = clean[col].fillna("unknown")
    clean["diagnosis"] = clean["diagnosis"].fillna("unknown")
    for col in ["age", "height", "weight", "bmi", "children", "months"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    return clean
