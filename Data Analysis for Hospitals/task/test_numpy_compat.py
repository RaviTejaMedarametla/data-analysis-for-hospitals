from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from utils.numpy_compat import _resolve_integrate, integrate


class _NumPyV1Stub:
    @staticmethod
    def trapz(y, x):
        return "trapz-called"


class _NumPyV2Stub:
    @staticmethod
    def trapezoid(y, x):
        return "trapezoid-called"


class NumPyCompatTests(unittest.TestCase):
    def test_resolve_integrate_uses_trapezoid_when_available(self):
        fn = _resolve_integrate(_NumPyV2Stub())
        self.assertEqual(fn([], []), "trapezoid-called")

    def test_resolve_integrate_falls_back_to_trapz(self):
        fn = _resolve_integrate(_NumPyV1Stub())
        self.assertEqual(fn([], []), "trapz-called")

    def test_integrate_matches_linear_area(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        self.assertAlmostEqual(float(integrate(y, x)), 2.0, places=7)


if __name__ == "__main__":
    unittest.main()
