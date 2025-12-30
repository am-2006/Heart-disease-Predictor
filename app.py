import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

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
    .result-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        background-color: white;
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
st.title("❤️ Heart Disease Assessment")
st.markdown("AI-Powered Clinical Risk Screening System")
st.divider()


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
    # CRITICAL: Feature alignment matching CSV order
    # Order: Age, Sex, CP, Trestbps, Chol, FBS, RestECG, Thalach, Exang, Oldpeak, Slope, CA, Thal
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    
    # 1. Scaling (This transforms raw inputs into z-scores)
    features_scaled = scaler.transform(features)
    
    # 2. Prediction
    prediction = model.predict(features_scaled)
    probability = float(prediction[0][0])
    
    # 3. UI Display
    st.divider()
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if probability > 0.5:
            st.error(f"### High Risk Identified")
            st.write("The clinical data correlates strongly with heart disease patterns.")
        else:
            st.success(f"### Low Risk Identified")
            st.write("The clinical data suggests a low statistical risk of heart disease.")

    with res_col2:
        st.metric("Risk Score", f"{probability:.2%}")
        st.progress(probability)

    # Guidance
    st.info("**Next Step:** This assessment is for screening purposes. Please consult a qualified medical professional for a final diagnosis.")

st.divider()
st.caption(" AI Medical Diagnostic System")
