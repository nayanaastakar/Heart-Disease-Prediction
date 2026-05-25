import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# Page Configuration (remove default header effect)
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).resolve().parent / 'heart_rf_model.pkl'
model = None

if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as exc:
        st.error(f"Unable to load trained model: {exc}")

# --------------------------------------------------
# HARD OVERRIDE STREAMLIT DEFAULT HEADER
# --------------------------------------------------
st.markdown("""
<style>

/* Hide Streamlit default top padding/header gap */
header {visibility: hidden;}
footer {visibility: hidden;}
#MainMenu {visibility: hidden;}

/* ---------- DARK THEME COLORS ---------- */
:root {
    --bg-main: #020617;
    --bg-sidebar: #020617;
    --bg-card: #0f172a;
    --text-main: #e5e7eb;
    --text-muted: #94a3b8;
    --border-soft: rgba(255,255,255,0.08);
}

/* ---------- MAIN APP ---------- */
.stApp {
    background-color: var(--bg-main);
    color: var(--text-main);
}

/* ---------- FORCE TEXT COLOR ---------- */
label, span, p, div {
    color: var(--text-main) !important;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar);
    border-right: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] * {
    color: var(--text-main) !important;
}

/* ---------- TITLE CARD ---------- */
.title-card {
    background: linear-gradient(180deg, #0f172a, #020617);
    padding: 38px;
    border-radius: 22px;
    box-shadow: 0 30px 60px rgba(0,0,0,0.9);
    margin-bottom: 30px;
}
.main-title {
    font-size: 46px;
    font-weight: 900;
    text-align: center;
}
.main-title span {
    background: linear-gradient(90deg, #ff6b6b, #fca5a5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 18px;
    margin-top: 8px;
}

/* ---------- CARD ---------- */
.card {
    background: var(--bg-card);
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 24px 48px rgba(0,0,0,0.85);
    margin-bottom: 20px;
}

/* ---------- INPUTS ---------- */
input, textarea {
    background-color: #020617 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}

/* ---------- SELECTBOX ---------- */
[data-baseweb="select"] > div {
    background-color: #020617 !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-soft) !important;
}
[role="listbox"] {
    background-color: #020617 !important;
}
[role="option"] {
    color: var(--text-main) !important;
}

/* ---------- BUTTON ---------- */
div.stButton > button {
    background: linear-gradient(90deg, #ff6b6b, #fca5a5);
    color: #020617 !important;
    font-weight: 800;
    border-radius: 10px;
    padding: 10px 16px;
    border: none;
}

/* ---------- RESULT ---------- */
.result-high {
    background: rgba(255, 80, 80, 0.15);
    color: #fecaca !important;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #ff6b6b;
}
.result-low {
    background: rgba(34, 197, 94, 0.15);
    color: #bbf7d0 !important;
    padding: 18px;
    border-radius: 12px;
    border-left: 6px solid #22c55e;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CUSTOM TITLE (ONLY HEADER THAT EXISTS)
# --------------------------------------------------
st.markdown("""
<div class="title-card">
    <div class="main-title">
        ❤️ <span>Heart Disease Risk Predictor</span>
    </div>
    <div class="subtitle">
        AI-assisted clinical risk assessment system
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------
left, right = st.columns([2, 1])

# --------------------------------------------------
# LEFT COLUMN — MANUAL INPUT ONLY
# --------------------------------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Patient Details")

    c1, c2 = st.columns(2)

    with c1:
        age = st.slider(
            "Age",
            20, 80, 45,
            help="Age of the patient in years. Risk increases with age."
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"],
            help="Biological sex of the patient."
        )

        trestbps = st.slider(
            "Resting Blood Pressure (mm Hg)",
            80, 200, 120,
            help="Blood pressure measured at rest."
        )

    with c2:
        chol = st.slider(
            "Serum Cholesterol (mg/dl)",
            100, 400, 200,
            help="Cholesterol level in blood."
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"],
            help="High fasting blood sugar indicates diabetes risk."
        )

    with st.expander("➕ Optional Clinical Parameters"):
        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"],
            help="Chest pain occurring during physical exercise."
        )

        oldpeak = st.slider(
            "ST Depression (Oldpeak)",
            0.0, 6.0, 1.0,
            help="ST depression induced by exercise compared to rest."
        )

        ca = st.selectbox(
            "Number of Major Vessels (0–3)",
            [0, 1, 2, 3],
            help="Number of major blood vessels with blockage."
        )

    sex_code = 1 if sex == "Male" else 0
    fbs_code = 1 if fbs == "Yes" else 0
    exang_code = 1 if exang == "Yes" else 0

    if st.button("🔍 Predict Risk"):
        if model is None:
            st.error(
                "Trained model not found. Run `python back.py` first to generate heart_rf_model.pkl."
            )
        else:
            input_data = pd.DataFrame([
                {
                    'age': age,
                    'sex': sex_code,
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': fbs_code,
                    'exang': exang_code,
                    'oldpeak': oldpeak,
                    'ca': ca,
                }
            ])
            probability = model.predict_proba(input_data)[0][1]
            percentage = round(probability * 100, 2)

            if probability > 0.5:
                st.markdown(
                    f"<div class='result-high'>⚠️ High Risk of Heart Disease<br><b>Estimated Risk:</b> {percentage}%</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-low'>✅ Low Risk of Heart Disease<br><b>Estimated Risk:</b> {percentage}%</div>",
                    unsafe_allow_html=True
                )

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# RIGHT COLUMN — EXPLANATION
# --------------------------------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ What does this result mean?")
    st.write("""
• **Low Risk** → Lower likelihood of heart disease  
• **High Risk** → Higher likelihood; consult a doctor  
• **Risk %** → Model confidence, not a diagnosis  
    """)
    st.write("""
⚠️ **Disclaimer:**  
This tool is for educational and decision-support purposes only.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<p style='text-align:center;opacity:0.6;'>Built for Kaggle Royale • Educational Use Only</p>",
    unsafe_allow_html=True
)
