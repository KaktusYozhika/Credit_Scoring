"""FastAPI-сервис оценки риска серьёзной кредитной просрочки."""

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (
    CreditApplication, HealthResponse, ModelInfoResponse, PredictionResponse
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "versions"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
state: dict = {"pipeline": None, "meta": {}, "error": None}


def load_artifacts(version: str = MODEL_VERSION):
    """Загружаем Pipeline и метаданные выбранной версии."""
    path = ARTIFACTS / version
    try:
        # joblib содержит Python-объект. Читаем только свои, проверенные файлы.
        pipeline = joblib.load(path / "model.joblib")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("model_version") != version:
            raise ValueError("Версия метаданных не совпадает с ожидаемой")
        state.update(pipeline=pipeline, meta=meta, error=None)
    except Exception as exc:
        state.update(pipeline=None, meta={}, error=str(exc))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Кредитный скоринг - группа 7",
    description="Оценка риска просрочки 90+ дней в течение двух лет. Учебный проект.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Служебные"])
def health():
    loaded = state["pipeline"] is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "model_version": state["meta"].get("model_version", MODEL_VERSION),
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Служебные"])
def model_info():
    meta = state["meta"]
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {key: meta[key] for key in (
        "model_version", "data_version", "trained_at", "features", "target",
        "threshold", "metrics", "validation_size", "train_size", "source"
    )}


@app.post("/predict", response_model=PredictionResponse, tags=["Прогноз"])
def predict(application: CreditApplication):
    if state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    meta = state["meta"]
    # Признаки поступают по именам (а не в произвольном порядке JSON).
    values = application.model_dump(by_alias=True)
    row = {name: values[name] for name in meta["features"]}
    frame = pd.DataFrame([row], columns=meta["features"])
    score = float(state["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(meta["threshold"])
    flagged = int(score >= threshold)
    return {
        "probability": round(score, 6),
        "predicted_class": flagged,
        "threshold": threshold,
        "model_version": meta["model_version"],
        "recommendation": (
            "additional_review" if flagged else "regular_review"
        ),
    }


# Для тестов и прямого импорта приложения.
load_artifacts()
