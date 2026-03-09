import pandas as pd
from feature_engineering.features import build_features


def test_feature_columns_added():
    df = pd.DataFrame({"age": [20], "bmi": [26], "height": [170], "weight": [70], "children": [0], "months": [0]})
    out = build_features(df)
    assert {"age_range", "is_adult", "bmi_risk"}.issubset(out.columns)
