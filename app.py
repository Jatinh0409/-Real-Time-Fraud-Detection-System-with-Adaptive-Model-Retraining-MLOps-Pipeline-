"""
app.py
FastAPI service that serves the fraud detection model.

Run locally:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /health   -> liveness check
    POST /predict  -> returns fraud probability for a transaction
    GET  /metrics  -> returns last-recorded training metrics (for the resume demo)
"""

import json
import os
import time
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "../models/model.pkl")
METRICS_PATH = os.getenv("METRICS_PATH", "../models/metrics.json")

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring service",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    model = joblib.load(MODEL_PATH)


class Transaction(BaseModel):
    hour_of_day: int = Field(..., ge=0, le=23)
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(..., gt=0)
    oldbalanceOrg: float = Field(..., ge=0)
    newbalanceOrig: float = Field(..., ge=0)
    oldbalanceDest: float = Field(..., ge=0)
    newbalanceDest: float = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "hour_of_day": 14,
                "type": "TRANSFER",
                "amount": 18000.50,
                "oldbalanceOrg": 20000.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 500.0,
                "newbalanceDest": 18500.5,
            }
        }


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    latency_ms: float


def engineer_features(payload: Transaction) -> pd.DataFrame:
    row = payload.dict()
    row["balance_diff_orig"] = row["oldbalanceOrg"] - row["newbalanceOrig"]
    row["balance_diff_dest"] = row["newbalanceDest"] - row["oldbalanceDest"]
    row["orig_drained"] = int(row["newbalanceOrig"] == 0)
    row["amount_to_balance_ratio"] = row["amount"] / (row["oldbalanceOrg"] + 1)
    return pd.DataFrame([row])


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="No metrics file found yet — train the model first.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    X = engineer_features(transaction)
    proba = float(model.predict_proba(X)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        fraud_probability=round(proba, 4),
        is_fraud=proba >= 0.5,
        latency_ms=round(latency_ms, 2),
    )
