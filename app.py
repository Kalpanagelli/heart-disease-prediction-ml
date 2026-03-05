

import streamlit as st
import joblib
import numpy as np
import os

# -------------------------------
# Custom CSS to remove gaps
# -------------------------------
st.markdown("""
<style>
    /* Remove gap between markdown and caption */
    .stMarkdown {
        margin-bottom: 0px !important;
    }
    
    /* Remove gap between caption and input elements */
    .stCaption {
        margin-bottom: -15px !important;
        margin-top: -15px !important;
    }
    
    /* Adjust spacing for number inputs, select boxes, and radio buttons */
    .stNumberInput, .stSelectbox, .stRadio {
        margin-top: -20px !important;
    }
    
    /* Additional adjustments for consistent spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0rem;
    }
    
    /* Reduce the top padding of input labels */
    .stNumberInput > label, .stSelectbox > label, .stRadio > label {
        margin-bottom: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Page Title
# -------------------------------

st.title("Heart Disease Prediction Tool")
st.write("Enter patient clinical information to estimate heart disease risk.")

# -------------------------------
# Load Model
# -------------------------------

BASE_DIR = os.path.dirname(__file__)
pipeline_path = os.path.join(BASE_DIR, "logistic_pipeline.pkl")
pipeline = joblib.load(pipeline_path)

THRESHOLD = 0.5

# -------------------------------
# USER INPUT SECTION
# -------------------------------

st.header("Patient Information")

# Age
st.markdown("**Age of patient**")
st.caption("Patient age in years (Example: 45)")
age = st.number_input("", min_value=1, max_value=120, value=50, key="age")

# Sex
st.markdown("**Sex**")
st.caption("Biological sex of the patient")
sex = st.selectbox("", ["Female", "Male"], key="sex")
sex = 1 if sex == "Male" else 0

# Chest Pain
st.markdown("**Chest Pain Type (1–4)**")
st.caption("""
1 = Typical angina  
2 = Atypical angina  
3 = Non-anginal pain  
4 = Asymptomatic
""")
chest_pain = st.selectbox("", [1,2,3,4], key="cp")

# Blood Pressure
st.markdown("**Resting Blood Pressure (mm Hg)**")
st.caption("Blood pressure measured at rest (Example: 120)")
bp = st.number_input("", min_value=50, max_value=250, value=120, key="bp")

# Cholesterol
TRAIN_MIN_CHOL = 126
TRAIN_MAX_CHOL = 564

st.markdown("**Cholesterol (mg/dL)**")
st.caption("Total cholesterol level in blood (Example: 200)")
chol = st.number_input("", min_value=0, max_value=600, value=200, key="chol")

if chol < TRAIN_MIN_CHOL or chol > TRAIN_MAX_CHOL:
    st.warning(
        f"⚠️ Cholesterol is outside the training data range "
        f"({TRAIN_MIN_CHOL}-{TRAIN_MAX_CHOL}). Prediction reliability may decrease."
    )

# Fasting Blood Sugar
st.markdown("**Fasting Blood Sugar > 120 mg/dL**")
st.caption("Indicates if fasting blood sugar is above 120 mg/dL")
fbs = st.radio("", ["No", "Yes"], key="fbs")
fbs = 1 if fbs == "Yes" else 0

# EKG
st.markdown("**Resting ECG Results**")
st.caption("""
0 = Normal  
1 = ST-T wave abnormality  
2 = Left ventricular hypertrophy
""")
ekg = st.selectbox("", [0,1,2], key="ekg")

# Max Heart Rate
st.markdown("**Maximum Heart Rate Achieved**")
st.caption("Highest heart rate achieved during exercise test (Example: 150)")
max_hr = st.number_input("", min_value=60, max_value=250, value=150, key="hr")

# Exercise Angina
st.markdown("**Exercise Induced Angina**")
st.caption("Chest pain triggered by physical activity")
exercise_angina = st.radio("", ["No","Yes"], key="angina")
exercise_angina = 1 if exercise_angina == "Yes" else 0

# ST Depression
st.markdown("**ST Depression**")
st.caption("ST depression induced by exercise relative to rest (Example: 1.0)")
st_depression = st.number_input("", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="st")

# Slope
st.markdown("**Slope of Peak Exercise ST Segment**")
st.caption("""
1 = Upsloping  
2 = Flat  
3 = Downsloping
""")
slope = st.selectbox("", [1,2,3], key="slope")

# Vessels
st.markdown("**Number of Major Vessels Colored by Fluoroscopy**")
st.caption("Number of major blood vessels detected in imaging (0–3)")
vessels = st.selectbox("", [0,1,2,3], key="vessels")

# Thallium
st.markdown("**Thallium Stress Test Result**")
st.caption("""
3 = Normal  
6 = Fixed defect  
7 = Reversible defect
""")
thallium = st.selectbox("", [3,6,7], key="thal")

# -------------------------------
# PREDICT BUTTON
# -------------------------------

st.markdown("---")

if st.button("Predict Heart Disease Risk"):

    features = np.array([[age, sex, chest_pain, bp, chol, fbs, ekg,
                          max_hr, exercise_angina, st_depression,
                          slope, vessels, thallium]])

    prob = pipeline.predict_proba(features)[0][1]

    prediction = "Heart Disease: YES" if prob >= THRESHOLD else "Heart Disease: NO"

    st.subheader("Prediction Result")

    st.write(prediction)

    st.write(f"Risk Score: {prob:.2f}")

    st.progress(prob)

    # Risk interpretation
    if prob < 0.30:
        st.success("Low Risk")
    elif prob < 0.60:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")

    st.warning("⚠️ This tool is for educational purposes only and not a medical diagnosis.")

# -------------------------------
# MODEL INFORMATION
# -------------------------------

st.markdown("---")

st.subheader("About This Model")

st.write("""
Model: Logistic Regression  
Dataset: UCI Heart Disease Dataset  
Features Used: 13 clinical variables  
Evaluation Metric: ROC-AUC  
AUC Score: ~0.89  

This machine learning model estimates the probability of heart disease
based on patient clinical measurements.
""")
