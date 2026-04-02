# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle

# ===============================
# Load trained model
# ===============================
with open("machine_failure_model.pkl", "rb") as f:
    model = pickle.load(f)

# ===============================
# FastAPI app
# ===============================
app = FastAPI(
    title="Machine Failure Prediction API",
    description="Predict if a machine will fail within 7 days based on sensor readings",
    version="1.0"
)

# ===============================
# Request model
# ===============================
class MachineData(BaseModel):
    plant_location: str
    temperature: float
    vibration: float
    pressure: float
    humidity: float
    runtime_hours: float
    load_percentage: float
    maintenance_history: float

# ===============================
# Response model
# ===============================
class PredictionResponse(BaseModel):
    prediction: int  # 0 = no failure, 1 = failure
    message: str

# ===============================
# Root endpoint
# ===============================
@app.get("/")
def root():
    return {"message": "Welcome to the Machine Failure Prediction API!"}

# ===============================
# Prediction endpoint
# ===============================
@app.post("/predict", response_model=PredictionResponse)
def predict_failure(data: MachineData):
    # Convert incoming data to dataframe
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline
    try:
        pred = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    message = "Machine is likely to fail within 7 days" if pred == 1 else "Machine is unlikely to fail within 7 days"

    return {"prediction": int(pred), "message": message}