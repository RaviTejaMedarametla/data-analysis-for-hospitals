from __future__ import annotations

import math
import numpy as np


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) <= 1:
        return mean, std, mean
    z = 1.96 if confidence == 0.95 else 1.64
    margin = z * std / math.sqrt(len(arr))
    return mean, std, float(margin)
