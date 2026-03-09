from pathlib import Path

from ingestion.loader import load_hospital_data
from ingestion.versioning import hash_file


def test_load_has_expected_keys():
    data_dir = Path("Data Analysis for Hospitals/test")
    datasets = load_hospital_data(data_dir)
    assert set(datasets.keys()) == {"general", "prenatal", "sports"}


def test_hash_file_length():
    path = Path("Data Analysis for Hospitals/test/general.csv")
    assert len(hash_file(path)) == 64
