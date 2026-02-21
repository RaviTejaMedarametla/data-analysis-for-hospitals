from __future__ import annotations

from pathlib import Path
import pandas as pd


HOSPITAL_FILES = {
    "general": "general.csv",
    "prenatal": "prenatal.csv",
    "sports": "sports.csv",
}


def load_hospital_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for hospital, file_name in HOSPITAL_FILES.items():
        path = data_dir / file_name
        datasets[hospital] = pd.read_csv(path)
    return datasets


def merge_hospital_data(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    general_columns = datasets["general"].columns
    aligned = []
    for _, frame in datasets.items():
        local = frame.copy()
        local.columns = general_columns
        aligned.append(local)
    merged = pd.concat(aligned, ignore_index=True)
    if "Unnamed: 0" in merged.columns:
        merged = merged.drop(columns=["Unnamed: 0"])
    return merged
