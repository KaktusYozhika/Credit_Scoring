"""Проверка артефактов и версий (включая откат)."""
import json
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "artifacts" / "versions"


class ModelArtifactTests(unittest.TestCase):
    def test_both_versions_exist(self):
        for version in ("v0.9.0", "v1.0.0"):
            for filename in ("model.joblib", "meta.json", "etalon.json"):
                self.assertTrue((VERSIONS / version / filename).exists())

    def test_etalon_predictions(self):
        for version in ("v0.9.0", "v1.0.0"):
            folder = VERSIONS / version
            pipeline = joblib.load(folder / "model.joblib")
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            examples = json.loads((folder / "etalon.json").read_text(encoding="utf-8"))
            self.assertEqual(len(examples), 5)
            self.assertEqual(meta["model_version"], version)
            for sample in examples:
                probability = float(pipeline.predict_proba(pd.DataFrame([sample["features"]]))[0, 1])
                self.assertAlmostEqual(probability, sample["probability"], delta=1e-6)
                self.assertEqual(int(probability >= meta["threshold"]), sample["predicted_class"])

    def test_rollback_threshold(self):
        old = json.loads((VERSIONS / "v0.9.0/meta.json").read_text(encoding="utf-8"))
        new = json.loads((VERSIONS / "v1.0.0/meta.json").read_text(encoding="utf-8"))
        self.assertEqual(old["threshold"], 0.50)
        self.assertEqual(new["threshold"], 0.55)
        self.assertEqual(old["features"], new["features"])
        self.assertEqual(old["data_version"], new["data_version"])


if __name__ == "__main__":
    unittest.main()
