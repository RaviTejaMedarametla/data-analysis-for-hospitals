import pandas as pd
from preprocessing.cleaning import clean_hospital_data


def test_cleaning_gender_and_numeric():
    df = pd.DataFrame({"gender": ["male", None], "bmi": [None, "25"], "children": [None, 1], "months": [None, 2], "blood_test": [None, "t"], "ecg": [None, "f"], "ultrasound": [None, "f"], "mri": [None, "f"], "xray": [None, "f"], "diagnosis": [None, "cold"], "age": ["20", "30"], "height": [170, 160], "weight": [70, 55]})
    out = clean_hospital_data(df)
    assert out.loc[0, "gender"] == "m"
    assert out["bmi"].dtype.kind in "fi"
