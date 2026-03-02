from __future__ import annotations

import os
import random
import numpy as np
import platform
from dataclasses import asdict, is_dataclass


def _set_default_threading_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def set_global_seed(seed: int) -> None:
    _set_default_threading_env()
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def reproducibility_context(config: object) -> dict:
    config_payload = asdict(config) if is_dataclass(config) else dict(config)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed": int(config_payload.get("random_seed", 0)),
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }
