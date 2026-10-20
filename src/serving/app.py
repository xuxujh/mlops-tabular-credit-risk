from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline

from src.data.generator import CreditApplicantRecord
from src.models.trainer import load_or_train_pipeline
from src.monitoring.drift import detect_feature_drift

model_pipeline: Pipeline | None = None
reference_distributions: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model_pipeline, reference_distributions
    model_pipeline = load_or_train_pipeline()
    ref_path = Path("models/reference_data.json")
    if ref_path.exists():
        reference_distributions = json.loads(ref_path.read_text(encoding="utf-8"))
    yield
    model_pipeline = None


app = FastAPI(
    title="MLOps Tabular Credit Risk API",
    description="API de inferência de risco de crédito e monitoramento de Data Drift (Scikit-learn & SciPy)",
    version="1.0.0",
    lifespan=lifespan,
)


class CreditPredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str
    approved: bool


class DriftCheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_data: dict[str, list[float]]


@app.get("/health", status_code=200)
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "service": "mlops-credit-risk-api"}


@app.post("/predict", response_model=CreditPredictionResponse, status_code=200)
async def predict_credit_risk(payload: CreditApplicantRecord) -> CreditPredictionResponse:
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_or_train_pipeline()

    try:
        input_df = pd.DataFrame([payload.model_dump()])
        prob = float(model_pipeline.predict_proba(input_df)[0, 1])

        if prob < 0.25:
            tier = "LOW_RISK"
            approved = True
        elif prob < 0.55:
            tier = "MEDIUM_RISK"
            approved = True
        else:
            tier = "HIGH_RISK"
            approved = False

        return CreditPredictionResponse(
            default_probability=round(prob, 4),
            risk_tier=tier,
            approved=approved,
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prediction error: {err}") from err


@app.post("/monitoring/drift", status_code=200)
async def check_drift(payload: DriftCheckRequest) -> dict[str, Any]:
    global reference_distributions
    if not reference_distributions:
        load_or_train_pipeline()
        ref_path = Path("models/reference_data.json")
        if ref_path.exists():
            reference_distributions = json.loads(ref_path.read_text(encoding="utf-8"))

    return detect_feature_drift(
        reference_data=reference_distributions,
        current_data=payload.current_data,
        alpha=0.05,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.serving.app:app", host="0.0.0.0", port=8000, reload=False)