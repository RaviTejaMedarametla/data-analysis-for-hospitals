import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "cli.py"


class ResearchPipelineTest(unittest.TestCase):
    def _run_cli(self):
        out = subprocess.check_output([sys.executable, str(CLI), "run"], text=True)
        return json.loads(out)

    def test_deterministic_core_metrics(self):
        run1 = self._run_cli()
        run2 = self._run_cli()
        self.assertAlmostEqual(run1["predictive_metrics"]["risk_accuracy"], run2["predictive_metrics"]["risk_accuracy"], places=10)
        self.assertAlmostEqual(run1["predictive_metrics"]["risk_auc"], run2["predictive_metrics"]["risk_auc"], places=10)
        self.assertEqual(run1["dataset_manifest_files"], run2["dataset_manifest_files"])

    def test_artifact_outputs_exist(self):
        run = self._run_cli()
        cv = run["predictive_cv"]
        for key in ["fold_artifact", "reliability_artifact", "report_artifact", "runtime_metadata_artifact"]:
            path = cv[key] if key in cv else run[key]
            self.assertTrue(Path(path).exists(), msg=f"Missing artifact: {path}")


if __name__ == "__main__":
    unittest.main()
