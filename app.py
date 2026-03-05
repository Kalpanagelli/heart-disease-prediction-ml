

import streamlit as st
import joblib
import numpy as np
import os

# Title
st.title("Heart Disease Prediction Tool")
st.write("Enter patient details to predict heart disease risk.")


# ---- Load pipeline using relative path ----
BASE_DIR = os.path.dirname(__file__)
pipeline_path = os.path.join(BASE_DIR, "logistic_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

# Set threshold
THRESHOLD = 0.5  # selected threshold


# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=50,
    help="Patient's age in years."
)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"],
    help="Biological sex of the patient."
)
sex = 1 if sex == "Male" else 0

chest_pain = st.selectbox(
    "Chest Pain Type",
    [1, 2, 3, 4],
    help="Type of chest discomfort experienced (1–4 categories based on medical classification). \n"
         "1 = Typical Angina (heart-related pain), \n"
         "2 = Atypical Angina(Chest pain that is somewhat heart-related but doesn’t follow the classic angina pattern.), \n"
         "3 = Non-anginal Pain (not heart related), \n"
         "4 = Asymptomatic (no chest pain)"
)

bp = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=50,
    max_value=250,
    value=120,
    help="Blood pressure measured while resting."
)

chol = st.number_input(
    "Cholesterol (mg/dL)",
    min_value=0,
    max_value=600,
    value=200,
    help="Total cholesterol level in blood."
)

max_hr = st.number_input(
    "Max Heart Rate Achieved",
    min_value=60,
    max_value=250,
    value=150,
    help="Highest heart rate achieved during exercise test."
)

st_depression = st.number_input(
    "ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Change in ECG reading during exercise. Higher values may indicate reduced blood flow to heart."
)

fbs = st.radio(
    "Fasting Blood Sugar > 120 mg/dL?",
    ["No", "Yes"],
    help="Measured after fasting for 8 hours. >120 mg/dL may indicate elevated risk."
)
fbs = 1 if fbs == "Yes" else 0

ekg = st.selectbox(
    "EKG Results",
    [0, 1, 2],
    help="Electrocardiogram test result showing heart electrical activity.\n"
         "0 = Normal, \n"
         "1 = ST-T abnormality, \n"
         "2 = Left ventricular hypertrophy"
)

exercise_angina = st.radio(
    "Exercise Induced Angina?",
    ["No", "Yes"],
    help="Chest pain triggered by physical activity."
)
exercise_angina = 1 if exercise_angina == "Yes" else 0

slope = st.selectbox(
    "Slope of ST Segment",
    [1, 2, 3],
    help= "Pattern of the ST segment during peak exercise (used in heart diagnosis).\n"
         "1 = Upsloping, \n"
         "2 = Flat,\n"
         "3 = Downsloping"
)

vessels = st.selectbox(
    "Number of Major Vessels (0–3)",
    [0, 1, 2, 3],
    help="Number of major blood vessels visible in fluoroscopy imaging."
)

thallium = st.selectbox (
    "Thallium Stress Test Result",
    [3, 6, 7],
    help= "Result of a nuclear stress test that checks blood flow to heart muscles.\n"
         "3 = Normal, \n"
         "6 = Fixed Defect (no blood flow in some part), \n"
         "7 = Reversible Defect (reduced blood flow during exercise)"
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------


features = np.array([[age, sex, chest_pain, bp, chol, fbs, ekg,
                      max_hr, exercise_angina, st_depression,
                      slope, vessels, thallium]])

prob = pipeline.predict_proba(features)[0][1]
prediction = "Heart Disease: YES" if prob >= THRESHOLD else "Heart Disease: NO"

# ---- Risk Level Interpretation ----
if prob < 0.30:
    risk_level = "Low Risk 🟢"
elif prob < 0.70:
    risk_level = "Moderate Risk 🟡"
else:
    risk_level = "High Risk 🔴"

# ---------------------------------------------------
# Display Results
# ---------------------------------------------------

st.subheader("Prediction Result")
st.write(prediction)
st.write(f"Probability of heart disease: {prob:.2f}")
st.markdown(f"### Risk Level: **{risk_level}**")

st.progress(prob)

st.info(
    "This tool is for educational purposes only and is not a medical diagnosis.")