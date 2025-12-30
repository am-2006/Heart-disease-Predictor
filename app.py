import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HeartCare AI | Heart Disease Risk Assessment",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f6f8fb;
}
.main {
    background-color: #ffffff;
    padding: 2.5rem;
    border-radius: 12px;
}
.header-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
}
.header-subtitle {
    font-size: 15px;
    text-align: center;
    color: #6c757d;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 35px;
}
.card {
    background-color: #f9fafb;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e0e0e0;
}
.footer {
    text-align: center;
    font-size: 13px;
    color: #6c757d;
    margin-top: 60px;
}
.small-text {
    font-size: 14px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("heart_disease_model.h5")

# ---------------- HEADER ----------------
st.markdown("<div class='header-title'>❤️ HeartCare AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='header-subtitle'>AI-Powered Heart Disease Risk Screening Platform</div>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- INTRO SECTION ----------------
st.markdown("<div class='section-title'>About the Platform</div>", unsafe_allow_html=True)
st.write(
    """
    **HeartCare AI** is an artificial intelligence–based medical screening system designed to
    assist in the early identification of heart disease risk using clinical patient data.
    The platform is intended to support preliminary assessment and awareness,
    and **does not replace professional medical diagnosis**.
    """
)

# ---------------- HOW IT WORKS ----------------
st.markdown("<div class='section-title'>How HeartCare AI Works</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        "<div class='card'><b>Clinical Dataset</b><br>"
        "Trained on a heart disease dataset containing clinically relevant attributes such as age, "
        "blood pressure, cholesterol, heart rate, and ECG-related parameters.</div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        "<div class='card'><b>AI Model</b><br>"
        "Uses a Deep Neural Network (DNN) developed with TensorFlow and Keras for binary classification "
        "of heart disease risk.</div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        "<div class='card'><b>Web-Based Assessment</b><br>"
        "Provides an interactive and user-friendly interface for entering patient data and "
        "viewing prediction results.</div>",
        unsafe_allow_html=True
    )

st.info(
    "Note: The system is currently in a developmental phase. While functional, prediction inconsistencies "
    "may occur due to dataset limitations and ongoing model refinement."
)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.title("Patient Medical Details")

age = st.sidebar.number_input("Age", 20, 100, 67)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female")
cp = st.sidebar.slider("Chest Pain Type (0–3)", 0, 3, 3)
bp = st.sidebar.number_input("Blood Pressure (mm Hg)", 80, 250, 115)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 564)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
ekg = st.sidebar.slider("ECG Result (0–2)", 0, 2, 2)
max_hr = st.sidebar.number_input("Maximum Heart Rate", 60, 250, 160)
ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
st_depression = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.6)
slope = st.sidebar.slider("ST Segment Slope (0–2)", 0, 2, 2)
vessels = st.sidebar.slider("Number of Major Vessels (0–3)", 0, 3, 0)
thallium = st.sidebar.slider("Thallium Test Result (1–3)", 1, 3, 2)

user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekg,
                        max_hr, ex_angina, st_depression,
                        slope, vessels, thallium]])

# ---------------- PREDICTION ----------------
st.markdown("<div class='section-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)

if st.sidebar.button("Run Assessment"):

    prediction = model.predict(user_input)
    probability = float(prediction[0][0])

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        if probability > 0.5:
            st.error("⚠️ High Risk of Heart Disease Detected")
            st.write(
                "The system indicates a higher risk level based on the provided clinical parameters. "
                "This result should be reviewed by a qualified medical professional."
            )
        else:
            st.success("✅ Lower Risk of Heart Disease Detected")
            st.write(
                "The system indicates a lower risk level. Regular health monitoring and "
                "preventive care are still recommended."
            )

    with col_r2:
        st.metric("Risk Probability", f"{probability:.2f}")
        st.progress(probability)

# ---------------- GUIDANCE ----------------
st.markdown("<div class='section-title'>Health Guidance</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='card'>
    <b>Important:</b><br><br>
    • Maintain regular medical check-ups.<br>
    • Follow a heart-healthy diet and active lifestyle.<br>
    • Avoid smoking and excessive alcohol consumption.<br>
    • Monitor blood pressure, cholesterol, and blood sugar periodically.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- EMERGENCY & HELPLINE ----------------
st.markdown("<div class='section-title'>Emergency Support</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='card'>
    <b>If you experience chest pain, breathlessness, or dizziness:</b><br><br>
    📞 <b>National Emergency Number (India):</b> 112<br>
    📱 <b>State Health Helpline:</b> 104<br>
    🏥 Seek immediate medical attention.
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- HOSPITAL LOCATOR ----------------
st.markdown("<div class='section-title'>Hospital & Cardiac Care Locator</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='card'>Use the links below to locate nearby hospitals and specialized cardiac care centers.</div>",
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    st.link_button("🏥 Find Nearby Hospitals", "https://www.google.com/maps/search/nearby+hospitals")

with c2:
    st.link_button("❤️ Find Cardiac Care Centers", "https://www.google.com/maps/search/cardiology+hospital+near+me")

# ---------------- DISCLAIMER ----------------
st.warning(
    "Disclaimer: HeartCare AI is a developing AI-based screening tool intended for educational "
    "and preliminary assessment purposes only. It should not be used as a substitute for "
    "professional medical diagnosis or treatment."
)

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>"
    "HeartCare AI | AI-Powered Medical Diagnosis System (-A.H.M)"
    "</div>",
    unsafe_allow_html=True
)
