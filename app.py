import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered",
)

# --- STYLING ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #e63946;
        color: white;
        font-weight: bold;
        height: 3em;
    }
    .emergency-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #fff5f5;
        border: 2px solid #feb2b2;
        color: #c53030;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_clinical_assets():
    try:
        model = tf.keras.models.load_model("heart_disease_model.h5")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        return None, None

model, scaler = load_clinical_assets()

# --- HEADER ---
st.title("🫀 Heart Disease Risk Predictor")
st.markdown("AI-Powered Clinical Risk Screening System")
st.divider()

if model is None or scaler is None:
    st.error("🚨 **File Error:** Could not find `heart_disease_model.h5` or `scaler.pkl` in the repository.")
    st.stop()

# --- INPUT FORM ---
st.subheader("Patient Medical Profile")
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 1, 100, 50)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], 
                         format_func=lambda x: {1:"Typical Angina", 2:"Atypical Angina", 3:"Non-anginal", 4:"Asymptomatic"}[x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        restecg = st.selectbox("Resting ECG Result", [0, 1, 2])

    with col2:
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0)
        slope = st.selectbox("ST Slope", [1, 2, 3])
        ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thallium Test", [3, 6, 7], 
                           format_func=lambda x: {3:"Normal", 6:"Fixed Defect", 7:"Reversible Defect"}[x])

# --- PREDICTION ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Generate Risk Assessment"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = float(prediction[0][0])
    
    st.divider()
    
    # --- REQUIREMENT 2: SUNBURST/CIRCLE GAUGE ---
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk %", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#e63946" if probability > 0.5 else "#2a9d8f"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 100], 'color': '#ffebee'}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- REQUIREMENT 1: HIGH RISK ADDITIONS ---
    if probability > 0.5:
        st.error("### ⚠️ High Risk Identified")
        st.markdown(f"""
        <div class="emergency-card">
            <strong>🚨 EMERGENCY STEPS:</strong><br>
            1. <strong>Cardiac Helpline:</strong> Call 102 or 108 (National Medical Emergency).<br>
            2. <strong>Seek Immediate Care:</strong> Chest pain or breathlessness requires urgent attention.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("🏥 Locate Nearby Cardiac Hospitals", 
                       "https://www.google.com/maps/search/cardiology+hospital+near+me")
    else:
        st.success("### ✅ Low Risk Identified")
        st.info("Continue maintaining a healthy lifestyle. Regular check-ups are still advised.")

st.divider()
st.caption("Developed by Aman Kumar Choudhary | Disclaimer: For screening only. Not a medical diagnosis.")            2. <strong>Seek Immediate Care:</strong> Chest pain or breathlessness requires urgent attention.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.link_button("🏥 Locate Nearby Cardiac Hospitals", 
                       "https://www.google.com/maps/search/cardiology+hospital+near+me")
    else:
        st.success("### ✅ Low Risk Identified")
        st.info("Continue maintaining a healthy lifestyle. Regular check-ups are still advised.")


    # Guidance
    st.info("**Next Step:** This assessment is for screening purposes. Please consult a qualified medical professional for a final diagnosis.")

st.divider()
st.caption(" AI Medical Diagnostic System")
