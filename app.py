

import streamlit as st
import joblib
import numpy as np
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction Tool")
st.write("Enter patient clinical details to estimate the likelihood of heart disease.")

# ------------------------------
# Load Model
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
pipeline_path = os.path.join(BASE_DIR, "logistic_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

THRESHOLD = 0.5

# ------------------------------
# Input Layout (2 Columns)
# ------------------------------
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50
    )
    st.caption("Age of the patient in years. Example: 45")

    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )
    st.caption("Biological sex of the patient.")

    chest_pain = st.selectbox(
        "Chest Pain Type (1–4)",
        [1,2,3,4]
    )
    st.caption("1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic")

    bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120
    )
    st.caption("Blood pressure measured while resting. Example: 120")

    chol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=0,
        max_value=600,
        value=200
    )
    st.caption("Total cholesterol level in blood.")

with col2:

    fbs = st.radio(
        "Fasting Blood Sugar > 120 mg/dL",
        ["No", "Yes"]
    )
    st.caption("Indicates whether fasting blood sugar exceeds 120 mg/dL.")

    ekg = st.selectbox(
        "Resting ECG Results (0–2)",
        [0,1,2]
    )
    st.caption("Electrocardiogram results measuring heart electrical activity.")

    max_hr = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=250,
        value=150
    )
    st.caption("Highest heart rate achieved during exercise testing.")

    exercise_angina = st.radio(
        "Exercise-Induced Angina",
        ["No","Yes"]
    )
    st.caption("Chest pain triggered by physical activity.")

    st_depression = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    st.caption("ECG measurement indicating reduced blood flow during exercise.")

# ------------------------------
# Additional Inputs
# ------------------------------

slope = st.selectbox(
    "Slope of ST Segment (1–3)",
    [1,2,3]
)
st.caption("1=Upsloping, 2=Flat, 3=Downsloping ST segment during peak exercise.")

vessels = st.selectbox(
    "Number of Major Vessels (0–3)",
    [0,1,2,3]
)
st.caption("Number of major vessels visible in fluoroscopy imaging.")

thallium = st.selectbox(
    "Thallium Stress Test Result",
    [3,6,7]
)
st.caption("3=Normal, 6=Fixed defect, 7=Reversible defect.")

# ------------------------------
# Convert Inputs
# ------------------------------
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exercise_angina = 1 if exercise_angina == "Yes" else 0

# ------------------------------
# Predict Button
# ------------------------------
if st.button("🔍 Predict Heart Disease Risk"):

    features = np.array([[age, sex, chest_pain, bp, chol, fbs, ekg,
                          max_hr, exercise_angina, st_depression,
                          slope, vessels, thallium]])

    prob = pipeline.predict_proba(features)[0][1]

    prediction = "Heart Disease: YES" if prob >= THRESHOLD else "Heart Disease: NO"

    st.subheader("Prediction Result")

    if prob >= THRESHOLD:
        st.error(prediction)
    else:
        st.success(prediction)

    st.write(f"**Probability of Heart Disease:** {prob:.2f}")

    # ------------------------------
    # Confidence Interpretation
    # ------------------------------
    if prob < 0.30:
        st.info("Low predicted risk")
    elif prob < 0.60:
        st.warning("Moderate predicted risk")
    else:
        st.error("High predicted risk")

# ------------------------------
# Disclaimer
# ------------------------------
st.divider()
st.caption(
"This tool is for educational purposes only and should not be used as a medical diagnosis."
)