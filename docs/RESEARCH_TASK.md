# Research Task Definition

## Objective
Predict patient risk as a binary early-warning indicator derived from diagnosis labels (`appendicitis` or `pregnancy` as positive class).

## Dataset
Source CSV files: `general.csv`, `prenatal.csv`, `sports.csv` under `Data Analysis for Hospitals/test/`.
Schema includes demographics, vitals, test outcomes, diagnosis, and hospital metadata.

## Splits
Deterministic split controlled by `split_seed` in `config.py`.
Default split ratio: 75% train / 25% test.

## Metrics
Primary: accuracy, F1, AUC.
Secondary: latency (ms), p95 latency, and memory delta (RSS MB).
