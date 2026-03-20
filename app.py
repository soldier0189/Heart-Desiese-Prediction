from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from typing import Literal
import pickle

app = FastAPI()

with open("./artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

class UserInfo(BaseModel):
    age: int = Field(gt= 0, lt=120, description="Age of the patient")
    sex: Literal["M", "F"] = Field(description="Gender of the patient")
    chest_pain_type: Literal["ASY", "NAP", "ATA", "TA"] = Field(description= "Chest pain type")
    resting_bp: int = Field(gt=0, description="Resting bp of the patient")
    cholesterol: int = Field(gt=0, description="cholesterol level")
    fasting_bs: int 
    resting_ecg: Literal["Normal","LVH","ST"]
    max_hr: int = Field(gt=0,  description="max hr")
    exercise_angina: Literal["Y","N"] = Field(description="exercise angina yes or no")
    old_peak: float = Field(description="old peak")
    st_slope: Literal["Flat", "Up", "Down"] = Field(description="Enter the st_slope of the patient - flat, up, down")

@app.get("/")
def home():
    return {"message": "Heart Desiese Prediction system"}

@app.post("/predict")
def predict(data: UserInfo):
    sex = 0 if data.sex == "M" else 1

    chest_pain_map = {"ASY":0, "NAP":1, "ATA":2, "TA":3}
    chest_pain_type = chest_pain_map[data.chest_pain_type]

    ecg_map = {"Normal":0, "LVH":1, "ST":2}
    resting_ecg = ecg_map[data.resting_ecg]

    exercise_angina = 1 if data.exercise_angina == "Y" else 0

    st_slope_map = {"Flat":0, "Up":1, "Down":2}
    st_slope = st_slope_map[data.st_slope]

    input_data = pd.DataFrame([{
        "Age": data.age,
        "Sex": sex,
        "ChestPainType": chest_pain_type,
        "RestingBP": data.resting_bp,
        "Cholesterol": data.cholesterol,
        "FastingBS": data.fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": data.max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": data.old_peak,
        "ST_Slope": st_slope
    }])

    prediction = model.predict(input_data)
    print(prediction)
    return {"prediction": int(prediction[0])}


