import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="HeartCare AI",
    page_icon="❤️",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #edf2f7;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #e63946;
        color: white;
        font-weight: bold;
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- ASSET LOADING ----------------
@st.cache_resource
def load_assets():
    model_path = "heart_disease_model.h5"
    scaler_path = "scaler.pkl"
    
    # Check if files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    # Load Model and Scaler
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_assets()

# ---------------- HEADER ----------------
st.title("❤️ HeartCare Risk Assessment")
st.markdown("AI-powered clinical screening system")
st.divider()



# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📋 Patient Information")

age = st.sidebar.number_input("Age", 1, 120, 55)
sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.sidebar.selectbox("Resting ECG Result (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0)
slope = st.sidebar.selectbox("ST Slope (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thallium Test Result (1-3)", [1, 2, 3])

# ---------------- PREDICTION ----------------
if st.sidebar.button("Analyze Risk"):
    # 1. Arrange data in the exact order the model expects
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                            thalach, exang, oldpeak, slope, ca, thal]])
    
    # 2. SCALE THE DATA (Crucial fix for "All High Risk" bug)
    input_data_scaled = scaler.transform(input_data)
    
    # 3. Get Prediction
    prediction = model.predict(input_data_scaled)
    probability = float(prediction[0][0])
    
    # 4. Results UI
    st.subheader("Assessment Result")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if probability > 0.5:
            st.error(f"### High Risk Identified")
            st.write("The clinical profile indicates a high statistical probability of heart disease presence.")
        else:
            st.success(f"### Low Risk Identified")
            st.write("The clinical profile indicates a low statistical probability of heart disease presence.")
    
    with col2:
        st.metric("Probability", f"{probability:.2%}")
        st.progress(probability)

    # Guidance Card
    st.markdown("""<div class='card'><h4>Recommendation</h4>""" + 
        ("Consult a cardiologist for a thorough examination." if probability > 0.5 else "Maintain a heart-healthy lifestyle and regular screenings.") + 
        """</div>""", unsafe_allow_html=True)

st.divider()
st.caption("Disclaimer: This is a screening tool, not a medical diagnosis. Consult a doctor for medical advice.")
