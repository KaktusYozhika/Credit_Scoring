"""Проверяем, что переключение версии меняет решение на контрольной заявке."""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RollbackAPITests(unittest.TestCase):
    def test_model_version_switch_changes_prediction(self):
        command = (
            "import json; from fastapi.testclient import TestClient; "
            "from app.main import app; "
            "data=json.load(open('rollback_example.json',encoding='utf8')); "
            "c=TestClient(app); "
            "r=c.post('/predict',json=data); assert r.status_code==200,r.text; "
            "print(json.dumps(r.json()))"
        )
        predictions = {}
        for version in ("v0.9.0", "v1.0.0"):
            env = os.environ.copy()
            env["MODEL_VERSION"] = version
            result = subprocess.run(
                [sys.executable, "-c", command], cwd=ROOT, env=env,
                check=True, capture_output=True, text=True,
            )
            predictions[version] = json.loads(result.stdout.strip())
            self.assertEqual(predictions[version]["model_version"], version)
        self.assertEqual(predictions["v0.9.0"]["predicted_class"], 1)
        self.assertEqual(predictions["v1.0.0"]["predicted_class"], 0)
        self.assertEqual(predictions["v0.9.0"]["threshold"], 0.5)
        self.assertEqual(predictions["v1.0.0"]["threshold"], 0.55)
        self.assertAlmostEqual(predictions["v0.9.0"]["probability"],
                               predictions["v1.0.0"]["probability"], places=6)


if __name__ == "__main__":
    unittest.main()
