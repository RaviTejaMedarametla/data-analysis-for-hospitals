from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from config import CONFIG
from ingestion.loader import load_hospital_data, merge_hospital_data
from preprocessing.cleaning import clean_hospital_data
from feature_engineering.features import build_features


def legacy_eda() -> None:
    datasets = load_hospital_data(CONFIG.data_dir)
    merged = merge_hospital_data(datasets)
    cleaned = clean_hospital_data(merged)
    df = build_features(cleaned)

    age_range_counts = df["age_range"].value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    age_range_counts.plot(kind="bar", color="skyblue")
    plt.title("Age Distribution of Patients")
    plt.xlabel("Age Range")
    plt.ylabel("Number of Patients")
    plt.show()

    diagnosis_counts = df["diagnosis"].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(diagnosis_counts, labels=diagnosis_counts.index, autopct="%1.1f%%", startangle=140)
    plt.title("Most Common Diagnosis")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x="hospital", y="height", data=df, inner="quartile")
    plt.title("Height Distribution by Hospitals")
    plt.show()

    most_common_age_range = age_range_counts.idxmax()
    most_common_diagnosis = diagnosis_counts.idxmax()
    print(f"The answer to the 1st question: {most_common_age_range}")
    print(f"The answer to the 2nd question: {most_common_diagnosis}")
    print(
        "The answer to the 3rd question: The gap and dual peaks are expected due to hospital specialization "
        "(sports-focused adults vs mixed-age general/prenatal populations)."
    )


if __name__ == "__main__":
    legacy_eda()
