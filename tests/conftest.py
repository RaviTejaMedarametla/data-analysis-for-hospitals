import sys
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1] / "Data Analysis for Hospitals" / "task"
sys.path.insert(0, str(TASK_DIR))
