import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>

/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1581595219315-a187dd40c322?auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Glass effect card */
.glass {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 30px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}

/* Title */
.title {
    font-size:48px;
    font-weight:700;
    text-align:center;
    color:white;
}

.subtitle {
    font-size:20px;
    text-align:center;
    color:#f1f1f1;
    margin-bottom:30px;
}

/* Result styles */
.good {
    background:#1b5e20;
    color:white;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
}

.bad {
    background:#b71c1c;
    color:white;
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
}

/* Button animation */
.stButton>button {
    background: linear-gradient(45deg,#ff416c,#ff4b2b);
    color:white;
    border:none;
    border-radius:8px;
    padding:12px 25px;
    font-size:16px;
    transition:0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow:0 5px 15px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>❤️ AI Heart Disease Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning Powered Healthcare Assistant</div>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("heart_model.pkl")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

# ---------------- INPUT PANEL ----------------
with col1:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("Patient Information")

    age = st.slider("Age", 20, 100, 40)

    BMI = st.number_input("BMI", 10.0, 50.0, 25.0)

    sysBP = st.slider("Systolic Blood Pressure", 80, 200, 120)

    glucose = st.slider("Glucose Level", 50, 400, 90)

    heartRate = st.slider("Heart Rate", 40, 150, 70)

    smoker = st.selectbox("Current Smoker", [0,1])

    cigsPerDay = st.slider("Cigarettes Per Day", 0, 50, 0)

    predict = st.button("Predict Heart Disease Risk")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- RESULT PANEL ----------------
with col2:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    st.subheader("Prediction Result")

    if predict:

        data = np.array([[age, smoker, cigsPerDay, sysBP, glucose, BMI, heartRate]])

        prediction = model.predict(data)

        if prediction[0] == 1:

            st.markdown("<div class='bad'>⚠️ High Risk of Heart Disease</div>", unsafe_allow_html=True)

        else:

            st.markdown("<div class='good'>✅ Low Risk of Heart Disease</div>", unsafe_allow_html=True)

        st.balloons()

    else:
        st.info("Enter patient details and click the prediction button.")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center style='color:white'>AI Healthcare App • Built with Streamlit & Machine Learning</center>", unsafe_allow_html=True)

