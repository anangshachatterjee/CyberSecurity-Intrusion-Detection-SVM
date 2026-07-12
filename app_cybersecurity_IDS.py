import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# ------------------------------------
# Load Saved Model
# ------------------------------------
model = joblib.load("linear_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------------
# Create FastAPI
# ------------------------------------
app = FastAPI(title="Cybersecurity Intrusion Detection API")

# ------------------------------------
# Input format
# ------------------------------------
class Traffic(BaseModel):
    Packet_Size: float
    Request_Frequency: float


@app.get("/")
def home():
    return {"message": "Cybersecurity Intrusion Detection API"}

# ------------------------------------
# Prediction API
# ------------------------------------
@app.post("/predict")
def predict(data: Traffic):
    sample = np.array([[data.Packet_Size, data.Request_Frequency]])

    sample = scaler.transform(sample)
    prediction = model.predict(sample)
    return {"Predicted Attack Type": prediction[0]}
    
