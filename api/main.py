"""
FastAPI Churn Prediction Service
-----------------------------------
Serves the trained XGBoost churn model over HTTP. Loads the model once
at startup (not per-request) via ChurnPredictor.
"""
import sys
import os

# Allow importing from src/ since api/ is a sibling folder, not a subfolder of src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from predict import ChurnPredictor
from api.schemas import CustomerData, PredictionResponse

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts customer churn risk using a tuned XGBoost model.",
    version="1.0.0",
)

# Loaded once at startup, reused across all requests
predictor: ChurnPredictor | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = ChurnPredictor()


@app.get("/")
def root():
    return {"status": "ok", "message": "Telecom Churn Prediction API is running."}


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        raw_data = customer.model_dump()
        result = predictor.predict(raw_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    risk_label = "High Risk" if result["churn_probability"] >= 0.6 else (
        "Medium Risk" if result["churn_probability"] >= result["threshold_used"] else "Low Risk"
    )

    return PredictionResponse(
        customerID=customer.customerID,
        churn_prediction=result["churn_prediction"],
        churn_probability=result["churn_probability"],
        threshold_used=result["threshold_used"],
        risk_label=risk_label,
    )