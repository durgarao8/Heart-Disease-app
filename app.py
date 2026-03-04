import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="❤️ Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# Custom CSS - Dark Theme with Dynamic Effects
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

    /* ---- Root & Background ---- */
    :root {
        --black: #0a0a0a;
        --dark: #111111;
        --card: #181818;
        --border: #2a2a2a;
        --red: #e63946;
        --red-glow: rgba(230, 57, 70, 0.4);
        --green: #2ecc71;
        --green-glow: rgba(46, 204, 113, 0.3);
        --text: #e8e8e8;
        --muted: #888888;
        --light-gray: #c0c0c0;
    }

    /* Animated background */
    .stApp {
        background: var(--black);
        background-image:
            radial-gradient(ellipse at 10% 20%, rgba(230, 57, 70, 0.07) 0%, transparent 50%),
            radial-gradient(ellipse at 90% 80%, rgba(46, 204, 113, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.02) 0%, transparent 70%);
        font-family: 'Rajdhani', sans-serif;
        color: var(--text);
    }

    /* Pulsing heartbeat line at very top */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, var(--red), var(--red), transparent);
        animation: scanline 3s ease-in-out infinite;
        z-index: 9999;
    }

    @keyframes scanline {
        0%, 100% { opacity: 0.3; transform: scaleX(0.3); }
        50% { opacity: 1; transform: scaleX(1); }
    }

    /* Floating particles via pseudo layer */
    .main > div {
        position: relative;
    }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }

    /* ---- Typography ---- */
    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        letter-spacing: 2px;
    }

    /* ---- Title Area ---- */
    .hero-title {
        text-align: center;
        padding: 2.5rem 0 1.5rem;
        position: relative;
    }
    .hero-title h1 {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow:
            0 0 20px rgba(230, 57, 70, 0.8),
            0 0 40px rgba(230, 57, 70, 0.4),
            0 0 80px rgba(230, 57, 70, 0.2);
        animation: pulse-text 2.5s ease-in-out infinite;
    }
    @keyframes pulse-text {
        0%, 100% { text-shadow: 0 0 20px rgba(230,57,70,0.8), 0 0 40px rgba(230,57,70,0.4); }
        50% { text-shadow: 0 0 30px rgba(230,57,70,1), 0 0 60px rgba(230,57,70,0.6), 0 0 100px rgba(230,57,70,0.3); }
    }
    .hero-title p {
        font-family: 'Rajdhani', sans-serif;
        color: var(--light-gray);
        font-size: 1.15rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 0.5rem;
        opacity: 0.8;
    }

    /* ---- ECG Animation ---- */
    .ecg-container {
        width: 100%;
        overflow: hidden;
        height: 60px;
        margin: 1rem 0;
        position: relative;
    }
    .ecg-line {
        width: 200%;
        height: 60px;
        animation: ecg-scroll 4s linear infinite;
    }
    @keyframes ecg-scroll {
        0% { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }

    /* ---- Cards / Sections ---- */
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--red), transparent);
        animation: shimmer 3s ease-in-out infinite;
    }
    @keyframes shimmer {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    .section-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.7rem;
        letter-spacing: 4px;
        text-transform: uppercase;
        color: var(--red);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
    }

    /* ---- Inputs ---- */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background: #1e1e1e !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: var(--red) !important;
        box-shadow: 0 0 10px rgba(230, 57, 70, 0.2) !important;
    }
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        border-color: var(--red) !important;
        box-shadow: 0 0 15px rgba(230, 57, 70, 0.35) !important;
        outline: none !important;
    }

    /* Labels */
    label, .stSelectbox label, .stNumberInput label {
        color: var(--light-gray) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }

    /* ---- Predict Button ---- */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #b71c1c, #e63946) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 1rem 2rem !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        cursor: pointer !important;
        box-shadow: 0 4px 25px rgba(230, 57, 70, 0.4) !important;
        transition: all 0.3s ease !important;
        animation: btn-glow 2.5s ease-in-out infinite;
    }
    @keyframes btn-glow {
        0%, 100% { box-shadow: 0 4px 25px rgba(230,57,70,0.4); }
        50% { box-shadow: 0 4px 40px rgba(230,57,70,0.8), 0 0 60px rgba(230,57,70,0.3); }
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 8px 40px rgba(230, 57, 70, 0.7) !important;
    }
    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
    }

    /* ---- Alert boxes ---- */
    .stAlert {
        border-radius: 12px !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        animation: result-appear 0.5s ease forwards;
    }
    @keyframes result-appear {
        from { opacity: 0; transform: translateY(10px) scale(0.97); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ---- Divider ---- */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #333, transparent) !important;
        margin: 1.5rem 0 !important;
    }

    /* ---- Stats bar at top ---- */
    .stat-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        padding: 1rem 0;
        border-top: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--red);
    }
    .stat-label {
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--muted);
        margin-top: 2px;
    }

    /* ---- Caption ---- */
    .stCaption {
        color: var(--muted) !important;
        text-align: center !important;
        font-family: 'Rajdhani', sans-serif !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
    }

    /* ---- Scrollbar ---- */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--black); }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--red); }

    /* Floating ECG dots animation */
    @keyframes float-up {
        0% { transform: translateY(0px) scale(1); opacity: 0.6; }
        50% { transform: translateY(-15px) scale(1.1); opacity: 1; }
        100% { transform: translateY(0px) scale(1); opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="hero-title">
    <h1>❤️ CardioScan AI</h1>
    <p>🩺 10-Year Coronary Heart Disease Risk Assessment System 🩺</p>
</div>

<!-- Animated ECG SVG -->
<div class="ecg-container">
    <svg class="ecg-line" viewBox="0 0 1200 60" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
        <polyline
            fill="none"
            stroke="#e63946"
            stroke-width="2"
            opacity="0.7"
            points="0,30 80,30 100,30 110,5 120,55 130,10 140,50 150,30 230,30
                    310,30 320,30 330,5 340,55 350,10 360,50 370,30 450,30
                    530,30 540,30 550,5 560,55 570,10 580,50 590,30 670,30
                    750,30 760,30 770,5 780,55 790,10 800,50 810,30 890,30
                    970,30 980,30 990,5 1000,55 1010,10 1020,50 1030,30 1110,30
                    1190,30 1200,30"
        />
        <polyline
            fill="none"
            stroke="#e63946"
            stroke-width="2"
            opacity="0.7"
            points="1200,30 1280,30 1300,30 1310,5 1320,55 1330,10 1340,50 1350,30 1430,30
                    1510,30 1520,30 1530,5 1540,55 1550,10 1560,50 1570,30 1650,30
                    1730,30 1740,30 1750,5 1760,55 1770,10 1780,50 1790,30 1870,30
                    1950,30 1960,30 1970,5 1980,55 1990,10 2000,50 2010,30 2090,30
                    2170,30 2180,30 2190,5 2200,55 2210,10 2220,50 2230,30 2310,30
                    2400,30"
        />
    </svg>
</div>

<!-- Stats Bar -->
<div class="stat-bar">
    <div class="stat-item">
        <div class="stat-value">💊 15</div>
        <div class="stat-label">Input Parameters</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">🧠 AI</div>
        <div class="stat-label">Powered Model</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">🫀 10yr</div>
        <div class="stat-label">Risk Window</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">⚡ Fast</div>
        <div class="stat-label">Real-time Scan</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load trained model
# -----------------------------
try:
    model = joblib.load("heart_model.pkl")
    st.markdown('<p style="color:#2ecc71;font-family:Rajdhani;font-size:0.85rem;letter-spacing:2px;text-align:center;">✅ ML MODEL LOADED SUCCESSFULLY</p>', unsafe_allow_html=True)
except:
    st.markdown('<p style="color:#e63946;font-family:Rajdhani;font-size:0.85rem;letter-spacing:2px;text-align:center;">⚠️ heart_model.pkl NOT FOUND — DEMO MODE</p>', unsafe_allow_html=True)
    model = None

st.markdown("---")

# -----------------------------
# Patient Inputs
# -----------------------------
st.markdown('<div class="section-label">🔬 PATIENT MEDICAL DIAGNOSTICS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="color:#888;font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">👤 Demographics & Lifestyle</p>', unsafe_allow_html=True)
    male = st.selectbox("⚥ Gender", [0, 1], help="0 = Female, 1 = Male", format_func=lambda x: "👩 Female" if x == 0 else "👨 Male")
    age = st.number_input("🎂 Age (years)", min_value=20, max_value=100, value=45)
    education = st.selectbox(
        "🎓 Education Level", [1, 2, 3, 4],
        help="1 = Some High School, 2 = High School/GED, 3 = Some College, 4 = College",
        format_func=lambda x: {1:"📚 Some High School", 2:"🏫 High School / GED", 3:"🎒 Some College", 4:"🎓 College Degree"}[x]
    )
    currentSmoker = st.selectbox("🚬 Current Smoker", [0, 1], format_func=lambda x: "🚭 Non-Smoker" if x == 0 else "🚬 Smoker")
    cigsPerDay = st.number_input("🔥 Cigarettes Per Day", min_value=0, max_value=50, value=0)
    BPMeds = st.selectbox("💊 Blood Pressure Medication", [0, 1], format_func=lambda x: "❌ Not Taking" if x == 0 else "✅ Currently Taking")
    prevalentStroke = st.selectbox("🧠 History of Stroke", [0, 1], format_func=lambda x: "✅ No History" if x == 0 else "⚠️ Stroke History")
    prevalentHyp = st.selectbox("🩸 Hypertension", [0, 1], format_func=lambda x: "✅ No Hypertension" if x == 0 else "⚠️ Hypertensive")

with col2:
    st.markdown('<p style="color:#888;font-size:0.75rem;letter-spacing:3px;text-transform:uppercase;margin-bottom:1rem;">🧪 Clinical Measurements</p>', unsafe_allow_html=True)
    diabetes = st.selectbox("🩺 Diabetes", [0, 1], format_func=lambda x: "✅ No Diabetes" if x == 0 else "⚠️ Diabetic")
    totChol = st.number_input("🧬 Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    sysBP = st.number_input("📈 Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120)
    diaBP = st.number_input("📉 Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80)
    BMI = st.number_input("⚖️ Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    heartRate = st.number_input("💓 Heart Rate (bpm)", min_value=40, max_value=200, value=75)
    glucose = st.number_input("🍬 Glucose Level (mg/dL)", min_value=40, max_value=400, value=85)

st.markdown("---")

# -----------------------------
# Predict Button
# -----------------------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🫀 RUN CARDIAC RISK SCAN", use_container_width=True)

st.markdown("---")

# -----------------------------
# Prediction Result
# -----------------------------
if predict_clicked:
    data = np.array([[male, age, education, currentSmoker, cigsPerDay,
                      BPMeds, prevalentStroke, prevalentHyp, diabetes,
                      totChol, sysBP, diaBP, BMI, heartRate, glucose]])

    if model is not None:
        prediction = model.predict(data)
        result = prediction[0]
    else:
        result = 0  # demo fallback

    if result == 1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1a0505,#2d0808);border:1px solid #e63946;
             border-radius:14px;padding:2rem;text-align:center;
             box-shadow:0 0 40px rgba(230,57,70,0.4);
             animation:result-appear 0.5s ease forwards;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">🚨</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#e63946;
                 font-weight:900;letter-spacing:3px;text-transform:uppercase;
                 text-shadow:0 0 20px rgba(230,57,70,0.8);">HIGH RISK DETECTED</div>
            <div style="font-family:'Rajdhani',sans-serif;color:#c0c0c0;margin-top:0.8rem;
                 font-size:1rem;letter-spacing:1px;">
                ⚠️ Elevated risk of Coronary Heart Disease within 10 years.<br>
                🏥 Immediate medical consultation strongly advised.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#021208,#062010);border:1px solid #2ecc71;
             border-radius:14px;padding:2rem;text-align:center;
             box-shadow:0 0 40px rgba(46,204,113,0.3);
             animation:result-appear 0.5s ease forwards;">
            <div style="font-size:3.5rem;margin-bottom:1rem;">💚</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#2ecc71;
                 font-weight:900;letter-spacing:3px;text-transform:uppercase;
                 text-shadow:0 0 20px rgba(46,204,113,0.8);">LOW RISK — CLEAR</div>
            <div style="font-family:'Rajdhani',sans-serif;color:#c0c0c0;margin-top:0.8rem;
                 font-size:1rem;letter-spacing:1px;">
                ✅ No significant risk of Coronary Heart Disease detected.<br>
                🥗 Maintain a healthy lifestyle and schedule regular check-ups.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1rem 0;border-top:1px solid #1e1e1e;">
    <p style="font-family:'Rajdhani',sans-serif;color:#444;font-size:0.75rem;
       letter-spacing:3px;text-transform:uppercase;">
        🤖 AI-Powered · 🏥 CardioScan System · 🔬 Machine Learning · ❤️ Heart Health
    </p>
    <p style="font-family:'Orbitron',monospace;color:#2a2a2a;font-size:0.65rem;letter-spacing:2px;">
        FOR INFORMATIONAL PURPOSES ONLY · NOT A MEDICAL DIAGNOSIS
    </p>
</div>
""", unsafe_allow_html=True)
