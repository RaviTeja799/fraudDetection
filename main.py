"""
main.py
=======
FastAPI service for real-time fraud prediction.

Run:
    uvicorn main:app --reload --port 8000

Then visit:
    http://127.0.0.1:8000/docs   (interactive Swagger UI)
"""

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from schemas import TransactionInput, PredictionOutput
from feature_engineering import engineer_features, align_to_training_columns

MODEL_PATH = "model/fraud_rf_model.pkl"
FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"
THRESHOLD_PATH = "model/decision_threshold.pkl"

# Holds loaded artifacts — populated at startup, avoids reloading on every request
ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load model artifacts once ---
    try:
        ml_artifacts["model"] = joblib.load(MODEL_PATH)
        ml_artifacts["feature_columns"] = joblib.load(FEATURE_COLUMNS_PATH)
        ml_artifacts["threshold"] = joblib.load(THRESHOLD_PATH)
        print("Model artifacts loaded successfully.")
        print("Feature columns:", ml_artifacts["feature_columns"])
        print("Decision threshold:", ml_artifacts["threshold"])
    except FileNotFoundError as e:
        print(f"ERROR: Could not load model artifacts — {e}")
        raise
    yield
    # --- Shutdown: nothing to clean up here ---
    ml_artifacts.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Random Forest fraud prediction service (Infosys Springboard project)",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the standalone frontend (opened as a local file or hosted anywhere)
# to call this API from the browser. Wide-open origins are fine for a
# portfolio/demo project — for a real production system this would be
# restricted to specific known domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Fraud Detection API is running. See /docs for usage.",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": "model" in ml_artifacts,
        "decision_threshold": ml_artifacts.get("threshold"),
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    if "model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # 1. Feature engineering — same logic as training
        engineered = engineer_features(transaction.model_dump())

        # 2. Align columns to exact training order
        X = align_to_training_columns(engineered, ml_artifacts["feature_columns"])

        # 3. Predict probability, apply tuned threshold (not sklearn's default 0.5)
        prob = ml_artifacts["model"].predict_proba(X)[:, 1][0]
        threshold = ml_artifacts["threshold"]
        prediction = int(prob >= threshold)

        return PredictionOutput(
            is_fraud_prediction=prediction,
            fraud_probability=round(float(prob), 4),
            decision_threshold=round(float(threshold), 4),
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")