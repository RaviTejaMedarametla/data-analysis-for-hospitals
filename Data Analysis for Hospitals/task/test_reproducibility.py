from __future__ import annotations

import os
import random
import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import CONFIG
from utils.reproducibility import reproducibility_context, set_global_seed


class ReproducibilityTests(unittest.TestCase):
    def test_seed_reproducibility_for_random_generators(self):
        set_global_seed(123)
        baseline_random = random.random()
        baseline_numpy = float(np.random.rand())

        set_global_seed(123)
        repeated_random = random.random()
        repeated_numpy = float(np.random.rand())

        self.assertEqual(baseline_random, repeated_random)
        self.assertEqual(baseline_numpy, repeated_numpy)

    def test_reproducibility_context_contains_required_keys(self):
        set_global_seed(CONFIG.random_seed)
        context = reproducibility_context(CONFIG)

        self.assertIn("python_version", context)
        self.assertIn("platform", context)
        self.assertEqual(context["seed"], CONFIG.random_seed)
        self.assertEqual(context["thread_env"]["PYTHONHASHSEED"], str(CONFIG.random_seed))

    def test_threading_env_defaults_are_set(self):
        set_global_seed(CONFIG.random_seed)
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("MKL_NUM_THREADS"), "1")
        self.assertEqual(os.environ.get("OPENBLAS_NUM_THREADS"), "1")


if __name__ == "__main__":
    unittest.main()
