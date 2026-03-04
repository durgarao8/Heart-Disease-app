import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("heart_model.pkl")

# -----------------------------
# Page Title
# -----------------------------
st.title("Heart Disease Prediction System")
st.write("Predict the risk of Coronary Heart Disease in 10 years")

st.divider()

st.subheader("Enter Patient Medical Details")

# -----------------------------
# Patient Inputs
# -----------------------------

col1, col2 = st.columns(2)

with col1:

    male = st.selectbox("Gender", [0,1], help="0 = Female, 1 = Male")

    age = st.number_input("Age", min_value=20, max_value=100)

    education = st.selectbox(
        "Education Level",
        [1,2,3,4],
        help="1 = Some High School, 2 = High School/GED, 3 = Some College, 4 = College"
    )

    currentSmoker = st.selectbox("Current Smoker", [0,1])

    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=50)

    BPMeds = st.selectbox("Blood Pressure Medication", [0,1])

    prevalentStroke = st.selectbox("History of Stroke", [0,1])

    prevalentHyp = st.selectbox("Hypertension", [0,1])


with col2:

    diabetes = st.selectbox("Diabetes", [0,1])

    totChol = st.number_input("Total Cholesterol", min_value=100, max_value=600)

    sysBP = st.number_input("Systolic Blood Pressure", min_value=80, max_value=250)

    diaBP = st.number_input("Diastolic Blood Pressure", min_value=40, max_value=150)

    BMI = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0)

    heartRate = st.number_input("Heart Rate", min_value=40, max_value=200)

    glucose = st.number_input("Glucose Level", min_value=40, max_value=400)


st.divider()

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Heart Disease Risk"):

    data = np.array([[male, age, education, currentSmoker, cigsPerDay,
                      BPMeds, prevalentStroke, prevalentHyp, diabetes,
                      totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease within 10 years")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.divider()

st.caption("Machine Learning based Heart Disease Prediction System")
