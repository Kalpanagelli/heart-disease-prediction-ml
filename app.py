# app.py
import streamlit as st
import joblib
import numpy as np

# Title
st.title("Heart Disease Prediction Tool")
st.write("Enter patient details to predict heart disease risk.")

# Load trained pipeline:

# Load pipeline
pipeline = joblib.load("C:\\Users\\kalpa\\OneDrive\\Data_Science_Projects\\Heart_Disease_Prediction_Project\\Models\\logistic_pipeline.pkl")

# Set threshold
THRESHOLD = 0.5  # your selected threshold

# Create User inputs

# Numeric Inputs
age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
bp = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
max_hr = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
st_depression = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Categorical Inputs
sex = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex == "Male" else 0

chest_pain = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
fbs = st.radio("FBS > 120?", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0

ekg = st.selectbox("EKG Results (0,1,2)", [0,1,2])
exercise_angina = st.radio("Exercise Induced Angina?", ["No","Yes"])
exercise_angina = 1 if exercise_angina == "Yes" else 0

slope = st.selectbox("Slope of ST (1–3)", [1,2,3])
vessels = st.selectbox("Number of Vessels Fluro (0–3)", [0,1,2,3])
thallium = st.selectbox("Thallium Test Result (1–3)", [1,2,3])

# Prepare feature and Predict

# Convert inputs to numpy array for pipeline
features = np.array([[age, sex, chest_pain, bp, chol, fbs, ekg, max_hr,
                      exercise_angina, st_depression, slope, vessels, thallium]])

# Predict probability
prob = pipeline.predict_proba(features)[0][1]

# Convert to Yes/No using threshold
prediction = "Heart Disease: YES" if prob >= THRESHOLD else "Heart Disease: NO"


# Display Results
st.subheader("Prediction Result")
st.write(prediction)
st.write(f"Probability of heart disease: {prob:.2f}")
st.warning("This tool is for educational purposes only. Not a medical diagnosis.")