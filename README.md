# Heart Disease Prediction Tool (End-to-End ML Project)

## Overview
This project is an end-to-end Machine Learning application that predicts the presence of heart disease based on patient clinical data.  
It covers the complete data science lifecycle — from data cleaning and exploratory data analysis (EDA) to model training, evaluation, and deployment as an interactive web application.

The final model is deployed using **Streamlit**, allowing users to input patient details and receive real-time predictions.

---

## Dataset
- Size: 271 records
- Features: 13 clinical attributes including age, blood pressure, cholesterol, ECG results, and exercise-induced indicators
- Target variable: **Heart Disease (1 = Presence, 0 = Absence)**

---

## Exploratory Data Analysis (EDA)
- Distribution analysis of numerical features
- Categorical feature impact on heart disease
- Correlation analysis
- Class balance inspection

---

## Model Development
Two models were trained and evaluated:

### 1️. Logistic Regression
- Feature scaling using StandardScaler
- Threshold tuning to balance precision and recall
- Final ROC-AUC: **0.8986**

### 2️. Random Forest (Hyperparameter Tuned)
- GridSearchCV for optimal parameters
- Final ROC-AUC: **0.8903**

**Final Model Selection:**  
Logistic Regression was selected due to:
- Higher ROC-AUC
- Better recall for heart disease cases
- Easier interpretability for medical-related use cases

---

## Machine Learning Pipeline
A Scikit-learn pipeline was used to:
- Prevent data leakage
- Ensure consistent preprocessing during inference
- Enable easy deployment

Pipeline components:
- StandardScaler
- Logistic Regression

---

## Web Application (Streamlit)
The trained pipeline is deployed as an interactive Streamlit app where users can:
- Enter patient clinical information
- Receive heart disease prediction
- View prediction probability

Note - *This tool is for educational purposes only and not a medical diagnosis.*

3. **Install dependencies**

pip install -r requirements.txt

**4. Run the Streamlit app**

python -m streamlit run app.py

**5. **Key Skills Demonstrated** **

Data Cleaning & Feature Engineering

Exploratory Data Analysis (EDA)

Logistic Regression & Random Forest

Hyperparameter & Threshold Tuning

ROC-AUC Evaluation

ML Pipeline Design

Model Deployment using Streamlit




👩‍💻 Author

Kalpana Gelli
Data Science | Machine Learning | Applied Analytics