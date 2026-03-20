import streamlit as st
import requests
import pandas as pd

URL = "http://127.0.0.1:8000/predict"

st.title("Heart Desiese Prediction: ")
age = st.number_input("Enter the Age")
sex = st.radio("Select Gender", ["M","F"])
chest_pain_type= st.selectbox("Type of chest pain", ["ASY", "NAP", "ATA", "TA"])
resting_bp = st.number_input("Resting BP")
cholesterol = st.number_input("Cholesterol Level")
fasting_bs = st.number_input("Fasting BS")
resting_ecg = st.selectbox("Resting ECG", ["Normal","LVH","ST"])
max_hr = st.number_input("Max HR")
exercise_angina = st.radio("Exercise Angina", ["Y", "N"])
old_peak = st.number_input("Old Peak")
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])


if st.button("Check Result"):
    post_data = {

    "age":age,
    "sex": sex,
    "chest_pain_type": chest_pain_type,
    "resting_bp": resting_bp,
    "cholesterol": cholesterol,
    "fasting_bs": fasting_bs,
    "resting_ecg": resting_ecg,
    "max_hr": max_hr,
    "exercise_angina": exercise_angina,
    "old_peak": old_peak,
    "st_slope": st_slope

}

    responce = requests.post(URL, json=post_data)
    if responce.status_code == 200:

        result = responce.json()
        if result["prediction"] == 1:
            st.warning("Patient has Heart Desiese")
        else:
            st.success("No desiese Detected")

