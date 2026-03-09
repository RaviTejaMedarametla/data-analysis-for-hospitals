#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
cd "Data Analysis for Hospitals/task"
python cli.py run > results/run.json
python cli.py manifest > results/manifest.json
python cli.py ablation > results/ablation.json
pytest -q
