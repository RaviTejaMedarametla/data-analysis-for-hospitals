from __future__ import annotations

import numpy as np


def _resolve_integrate(numpy_module: object = np):
    try:
        return numpy_module.trapezoid
    except AttributeError:
        return numpy_module.trapz


integrate = _resolve_integrate(np)

