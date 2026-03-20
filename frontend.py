import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("artifacts/model.pkl", "rb"))

st.title("Heart Disease Prediction ❤️")

# Inputs
age = st.number_input("Enter the Age")
sex = st.radio("Select Gender", ["M","F"])
chest_pain_type = st.selectbox("Type of chest pain", ["ASY", "NAP", "ATA", "TA"])
resting_bp = st.number_input("Resting BP")
cholesterol = st.number_input("Cholesterol Level")
fasting_bs = st.number_input("Fasting BS")
resting_ecg = st.selectbox("Resting ECG", ["Normal","LVH","ST"])
max_hr = st.number_input("Max HR")
exercise_angina = st.radio("Exercise Angina", ["Y", "N"])
old_peak = st.number_input("Old Peak")
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])

# Encode inputs (VERY IMPORTANT)
def preprocess():
    return pd.DataFrame([{
        "Age": age,
        "Sex": 0 if sex == "M" else 1,
        "ChestPainType": {"ASY":0, "NAP":1, "ATA":2, "TA":3}[chest_pain_type],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": {"Normal":0, "LVH":1, "ST":2}[resting_ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": 1 if exercise_angina == "Y" else 0,
        "Oldpeak": old_peak,
        "ST_Slope": {"Flat":0, "Up":1, "Down":2}[st_slope]
    }])

# Prediction
if st.button("Check Result"):
    input_df = preprocess()
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")