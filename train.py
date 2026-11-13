"""Практика 4: обучение кредитного скоринга и сохранение артефактов.

Запуск: python train.py
Модель полностью повторяет настройки практики 3 (разделение 70/15/15,
медианная импутация + индикаторы пропусков, масштабирование, логрегрессия).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "cs-training.csv"
VERSIONS_DIR = ROOT / "artifacts" / "versions"
TARGET = "SeriousDlqin2yrs"
SEED = 42
# Предыдущая и текущая конфигурации одного обученного Pipeline.
VERSIONS = {"v0.9.0": 0.50, "v1.0.0": 0.55}


def load_data(path: Path = DATA_FILE) -> tuple[pd.DataFrame, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Файл данных не найден: {path}")
    table = pd.read_csv(path)
    table = table.drop(columns=["Unnamed: 0"], errors="ignore")
    if TARGET not in table.columns:
        raise ValueError(f"Отсутствует целевая переменная {TARGET}")
    if table[TARGET].isna().any():
        raise ValueError("Есть пропуски в целевой переменной")
    duplicates = int(table.duplicated().sum())
    table = table.drop_duplicates().reset_index(drop=True)
    return table, duplicates


def split_data(table: pd.DataFrame):
    # Разделение идентично практике 3: validation и test по 15%.
    first, test = train_test_split(
        table, test_size=0.15, stratify=table[TARGET], random_state=SEED
    )
    train, validation = train_test_split(
        first, test_size=0.15 / 0.85,
        stratify=first[TARGET], random_state=SEED,
    )
    return train, validation, test


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("impute_and_scale", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])),
        ("logistic", LogisticRegression(
            solver="liblinear", max_iter=1000,
            class_weight="balanced", random_state=SEED,
        )),
    ])


def measure(y_true, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = (scores >= threshold).astype(int)
    return {
        "pr_auc_ap": round(float(average_precision_score(y_true, scores)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 5),
        "precision": round(float(precision_score(y_true, labels, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, labels, zero_division=0)), 5),
        "review_fraction": round(float(labels.mean()), 5),
    }


def value_to_json(value):
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def make_etalon(pipeline, validation: pd.DataFrame, features: list[str], threshold: float):
    # Для примеров используем именно валидные по API записи.
    good = validation[
        validation["age"].between(18, 110)
        & (validation["NumberOfTime30-59DaysPastDueNotWorse"] <= 95)
        & (validation["NumberOfTimes90DaysLate"] <= 95)
        & (validation["NumberOfTime60-89DaysPastDueNotWorse"] <= 95)
    ].head(5)
    if len(good) != 5:
        raise ValueError("Не удалось выбрать пять корректных эталонных записей")
    result = []
    for _, row in good.iterrows():
        features_values = {name: value_to_json(row[name]) for name in features}
        p = float(pipeline.predict_proba(pd.DataFrame([features_values]))[0, 1])
        result.append({
            "features": features_values,
            "probability": round(p, 6),
            "predicted_class": int(p >= threshold),
        })
    return result


def main():
    table, duplicate_count = load_data()
    train, validation, test = split_data(table)
    features = [x for x in table.columns if x != TARGET]
    pipeline = make_pipeline()
    pipeline.fit(train[features], train[TARGET])
    scores = pipeline.predict_proba(validation[features])[:, 1]

    data_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()[:12]
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base_rule = (validation["NumberOfTimes90DaysLate"] > 0).astype(int).to_numpy()
    base_ap = round(float(average_precision_score(validation[TARGET], base_rule)), 5)

    # Найдём заявку, для которой решение действительно меняется после отката:
    # при пороге 0.50 класс 1, а при пороге 0.55 класс 0.
    candidates = validation.loc[(scores >= 0.50) & (scores < 0.55)]
    candidates = candidates[
        candidates["age"].between(18, 110)
        & (candidates["NumberOfTime30-59DaysPastDueNotWorse"] <= 95)
        & (candidates["NumberOfTimes90DaysLate"] <= 95)
        & (candidates["NumberOfTime60-89DaysPastDueNotWorse"] <= 95)
    ]
    if candidates.empty:
        raise ValueError("Не найдена корректная заявка для демонстрации отката")
    rollback_sample = {name: value_to_json(candidates.iloc[0][name]) for name in features}
    (ROOT / "rollback_example.json").write_text(
        json.dumps(rollback_sample, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for version, threshold in VERSIONS.items():
        target_dir = VERSIONS_DIR / version
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, target_dir / "model.joblib")
        meta = {
            "model_version": version,
            "data_version": f"cs-training.csv:sha256:{data_hash}",
            "source": "Give Me Some Credit (Kaggle), cs-training.csv",
            "trained_at": trained_at,
            "target": TARGET,
            "features": features,
            "threshold": threshold,
            "train_size": len(train),
            "validation_size": len(validation),
            "test_size_not_used_for_selection": len(test),
            "removed_exact_duplicates": duplicate_count,
            "validation_class_1_fraction": round(float(validation[TARGET].mean()), 5),
            "baseline_pr_auc_ap": base_ap,
            "metrics": measure(validation[TARGET], scores, threshold),
            "note": "Это учебная модель. Баллы модели не откалиброваны для реального банковского решения.",
        }
        (target_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (target_dir / "etalon.json").write_text(
            json.dumps(make_etalon(pipeline, validation, features, threshold),
                       ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(version, "PR-AUC:", meta["metrics"]["pr_auc_ap"],
              "recall:", meta["metrics"]["recall"],
              "доля проверки:", meta["metrics"]["review_fraction"])

    print("Исходных строк:", len(table) + duplicate_count)
    print("После удаления совпадений:", len(table))
    print("Удалено совпадений:", duplicate_count)
    print("Train / validation / test:", len(train), len(validation), len(test))
    print("Артефакты:", VERSIONS_DIR)


if __name__ == "__main__":
    main()
