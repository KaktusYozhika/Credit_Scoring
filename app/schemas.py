"""Описание входных данных и ответа API."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class CreditApplication(BaseModel):
    """Десять признаков заёмщика из Give Me Some Credit.

    Доход и количество иждивенцев необязательны: они имеют пропуски
    и при обучении обрабатываются внутри Pipeline.
    """

    RevolvingUtilizationOfUnsecuredLines: float = Field(ge=0)
    age: int = Field(ge=18, le=110)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(ge=0, le=95, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(ge=0)
    MonthlyIncome: Optional[float] = Field(default=None, ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(ge=0)
    NumberOfTimes90DaysLate: int = Field(ge=0, le=95)
    NumberRealEstateLoansOrLines: int = Field(ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(ge=0, le=95, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: Optional[float] = Field(default=None, ge=0)

    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False,
        json_schema_extra={
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.45,
                "age": 39,
                "NumberOfTime30-59DaysPastDueNotWorse": 1,
                "DebtRatio": 0.28,
                "MonthlyIncome": 5200.0,
                "NumberOfOpenCreditLinesAndLoans": 7,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2,
            }
        },
    )


class PredictionResponse(BaseModel):
    probability: float = Field(ge=0, le=1)
    predicted_class: int = Field(ge=0, le=1)
    threshold: float
    model_version: str
    recommendation: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    data_version: str
    trained_at: str
    features: list[str]
    target: str
    threshold: float
    metrics: dict[str, float]
    validation_size: int
    train_size: int
    source: str
