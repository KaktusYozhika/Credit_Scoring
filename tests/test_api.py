"""Проверка REST API: 200, 422, служебные эндпоинты."""
import json
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

ROOT = Path(__file__).resolve().parent.parent
ETALON = json.loads((ROOT / "artifacts/versions/v1.0.0/etalon.json").read_text(encoding="utf-8"))


class APIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._context = TestClient(app)
        cls.client = cls._context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._context.__exit__(None, None, None)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["model_version"], "v1.0.0")

    def test_model_info(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target"], "SeriousDlqin2yrs")
        self.assertEqual(response.json()["threshold"], 0.55)
        self.assertEqual(len(response.json()["features"]), 10)
        self.assertIn("pr_auc_ap", response.json()["metrics"])

    def test_predict_valid(self):
        response = self.client.post("/predict", json=ETALON[0]["features"])
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["predicted_class"], ETALON[0]["predicted_class"])
        self.assertAlmostEqual(body["probability"], ETALON[0]["probability"], delta=1e-6)
        self.assertIn(body["recommendation"], ("regular_review", "additional_review"))
        self.assertGreaterEqual(body["probability"], 0)
        self.assertLessEqual(body["probability"], 1)

    def test_predict_missing_income_and_dependents(self):
        row = dict(ETALON[0]["features"])
        row.pop("MonthlyIncome", None)
        row.pop("NumberOfDependents", None)
        response = self.client.post("/predict", json=row)
        self.assertEqual(response.status_code, 200, response.text)

    def test_missing_required_field_is_422(self):
        row = dict(ETALON[0]["features"])
        row.pop("age")
        self.assertEqual(self.client.post("/predict", json=row).status_code, 422)

    def test_invalid_age_is_422(self):
        row = dict(ETALON[0]["features"], age=0)
        self.assertEqual(self.client.post("/predict", json=row).status_code, 422)

    def test_anomalous_late_code_is_422(self):
        row = dict(ETALON[0]["features"], NumberOfTimes90DaysLate=98)
        self.assertEqual(self.client.post("/predict", json=row).status_code, 422)

    def test_extra_field_is_422(self):
        row = dict(ETALON[0]["features"], imaginary_feature=9)
        self.assertEqual(self.client.post("/predict", json=row).status_code, 422)

    def test_docs_open(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)


if __name__ == "__main__":
    unittest.main()
