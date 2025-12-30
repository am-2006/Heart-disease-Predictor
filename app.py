import streamlit as st
import numpy as np
import tensorflow as tf

# Page configuration (light, professional)
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="centered"
)

# Load trained model
model = tf.keras.models.load_model("heart_disease_model.h5")

# Header
st.markdown(
    """
    <h2 style='text-align: center;'>Heart Disease Risk Prediction System</h2>
    <p style='text-align: center; color: grey;'>
    AI-assisted preliminary risk assessment (Prototype)
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Sidebar – Patient Input
st.sidebar.markdown("### Patient Information")

age = st.sidebar.number_input("Age", 20, 100, 67)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

cp = st.sidebar.slider("Chest Pain Type", 0, 3, 3)
bp = st.sidebar.number_input("Blood Pressure (mm Hg)", 80, 250, 115)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", 100, 600, 564)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
ekgr = st.sidebar.slider("ECG Result", 0, 2, 2)
max_hr = st.sidebar.number_input("Maximum Heart Rate", 60, 250, 160)
ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
st_depression = st.sidebar.number_input("ST Depression", 0.0, 10.0, 1.6)
slope = st.sidebar.slider("ST Segment Slope", 0, 2, 2)
vessels = st.sidebar.slider("Major Vessels (0–3)", 0, 3, 0)
thallium = st.sidebar.slider("Thallium Test Result", 1, 3, 2)

# Prepare input
user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekgr,
                        max_hr, ex_angina, st_depression,
                        slope, vessels, thallium]])

# Prediction button
if st.sidebar.button("Assess Risk"):

    prediction = model.predict(user_input)
    probability = prediction[0][0]

    st.markdown("### Assessment Result")

    if probability > 0.5:
        st.warning("⚠️ **High Risk Detected**")
        st.markdown(
            """
            This prediction suggests a **higher likelihood of heart disease**.
            Please consult a qualified medical professional for further evaluation.
            """
        )
    else:
        st.success("✅ **Low Risk Detected**")
        st.markdown(
            """
            This result indicates a **lower likelihood of heart disease**.
            Maintain a healthy lifestyle and regular medical check-ups.
            """
        )

    st.markdown(f"**Risk Probability:** `{probability:.2f}`")

# Footer / watermark
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 12px; color: grey;'>
    Developed by Aman Kumar Choudhary | Academic Project (Under Development)
    </p>
    """,
    unsafe_allow_html=True
)
