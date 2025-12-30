import streamlit as st
import numpy as np
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f7f9fc;
}
.main {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 10px;
}
.header-title {
    font-size: 36px;
    font-weight: 600;
    text-align: center;
}
.header-subtitle {
    font-size: 16px;
    text-align: center;
    color: #6c757d;
}
.section-title {
    font-size: 22px;
    font-weight: 500;
    margin-top: 30px;
}
.card {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
.footer {
    text-align: center;
    font-size: 13px;
    color: #6c757d;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("heart_disease_model.h5")

# ---------------- HEADER ----------------
st.markdown("<div class='header-title'>Heart Disease Risk Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>AI-powered clinical risk screening platform</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'><b>Clinical Focus</b><br>Uses patient medical parameters for risk estimation.</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><b>AI Engine</b><br>Deep learning model trained on clinical datasets.</div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'><b>Web Platform</b><br>Secure, browser-based assessment tool.</div>", unsafe_allow_html=True)

# ---------------- MAIN CONTENT ----------------
st.markdown("<div class='section-title'>Patient Risk Evaluation</div>", unsafe_allow_html=True)
st.write(
    "Enter patient medical information in the sidebar to generate a preliminary "
    "heart disease risk assessment. This system supports early screening and "
    "decision assistance."
)

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.title("Patient Details")

age = st.sidebar.number_input("Age", 20, 100, 67)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female")
cp = st.sidebar.slider("Chest Pain Type", 0, 3, 3)
bp = st.sidebar.number_input("Blood Pressure (mm Hg)", 80, 250, 115)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 564)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
ekg = st.sidebar.slider("ECG Result", 0, 2, 2)
max_hr = st.sidebar.number_input("Maximum Heart Rate", 60, 250, 160)
ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
st_depression = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.6)
slope = st.sidebar.slider("ST Segment Slope", 0, 2, 2)
vessels = st.sidebar.slider("Number of Major Vessels", 0, 3, 0)
thallium = st.sidebar.slider("Thallium Test Result", 1, 3, 2)

user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekg,
                        max_hr, ex_angina, st_depression,
                        slope, vessels, thallium]])

# ---------------- PREDICTION ----------------
if st.sidebar.button("Run Risk Assessment"):

    prediction = model.predict(user_input)
    probability = float(prediction[0][0])

    st.markdown("<div class='section-title'>Assessment Result</div>", unsafe_allow_html=True)

    result_col1, result_col2 = st.columns([2, 1])

    with result_col1:
        if probability > 0.5:
            st.error("High Risk of Heart Disease Detected")
            st.write(
                "The system indicates a higher likelihood of heart disease. "
                "This result should be reviewed by a qualified healthcare professional."
            )
        else:
            st.success("Low Risk of Heart Disease Detected")
            st.write(
                "The system indicates a lower likelihood of heart disease. "
                "Regular medical checkups are still recommended."
            )

    with result_col2:
        st.metric("Risk Probability", f"{probability:.2f}")
        st.progress(probability)

# ---------------- HEALTH GUIDANCE ----------------
st.markdown("<div class='section-title'>Personalized Health Guidance</div>", unsafe_allow_html=True)

if probability > 0.5:
    st.markdown(
        """
        <div class='card'>
        <b>Recommended Actions:</b><br><br>
        • Schedule a consultation with a certified cardiologist.<br>
        • Monitor blood pressure, cholesterol, and blood sugar regularly.<br>
        • Avoid smoking and limit alcohol consumption.<br>
        • Engage in light to moderate physical activity as advised by a doctor.<br>
        • Follow a heart-healthy diet rich in fruits, vegetables, and whole grains.
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class='card'>
        <b>Preventive Suggestions:</b><br><br>
        • Maintain a balanced diet and regular exercise routine.<br>
        • Attend periodic health check-ups.<br>
        • Manage stress through relaxation techniques.<br>
        • Maintain healthy sleep habits.<br>
        • Continue monitoring vital health indicators.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- HELPLINE SECTION ----------------
st.markdown("<div class='section-title'>Emergency Support & Helpline</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='card'>
    <b>If you are experiencing symptoms such as chest pain, breathlessness, or dizziness:</b><br><br>
    📞 <b>National Emergency Number (India):</b> 112<br>
    🏥 <b>Nearby Hospital / Cardiac Care Center</b><br>
    📱 <b>Health Helpline:</b> 104 (State Health Helpline)<br><br>
    </div>
    """,
    unsafe_allow_html=True
)
# ---------------- HOSPITAL LOCATOR ----------------
st.markdown("<div class='section-title'>Nearby Hospital Locator</div>", unsafe_allow_html=True)

st.markdown(
    """
    <div class='card'>
    Locate nearby hospitals and cardiac care centers using Google Maps.
    Click the button below to find medical facilities near your current location.
    </div>
    """,
    unsafe_allow_html=True
)

col_h1, col_h2 = st.columns(2)

with col_h1:
    st.link_button(
        "🏥 Find Nearby Hospitals",
        "https://www.google.com/maps/search/nearby+hospitals"
    )

with col_h2:
    st.link_button(
        "❤️ Find Cardiac Care Centers",
        "https://www.google.com/maps/search/cardiology+hospital+near+me"
    )
# ---------------- DISCLAIMER ----------------
st.warning(
    "Disclaimer: This platform is a developing AI-based screening system and "
    "is not intended to replace professional medical diagnosis."
)

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>"
    "AI-Powered Medical Diagnosis System (A.H.M)"
    "</div>",
    unsafe_allow_html=True
)
