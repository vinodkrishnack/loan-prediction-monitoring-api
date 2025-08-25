from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Dynamically resolve path relative to this file's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level up from app/
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.joblib")

model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    income: float
    credit_score: float
    gender: int
    employment_status: int

@app.post("/predict")
def predict(data: InputData):
    input_vector = np.array([[data.income, data.credit_score, data.gender, data.employment_status]])
    prediction = model.predict(input_vector)[0]
    return {"loan_approved": bool(prediction)}
