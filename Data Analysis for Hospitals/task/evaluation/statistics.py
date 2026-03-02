from __future__ import annotations

import math
import numpy as np


def confidence_interval(values: list[float], confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) <= 1:
        return mean, std, 0.0

    z_map = {0.90: 1.64, 0.95: 1.96, 0.99: 2.58}
    z = z_map.get(round(confidence, 2), 1.96)
    margin = z * std / math.sqrt(len(arr))
    return mean, std, float(margin)
