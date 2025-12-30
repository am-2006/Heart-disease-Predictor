import streamlit as st
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="🫀",
    layout="centered"
)

model = tf.keras.models.load_model("heart_disease_model.h5")

# ================= HEADER =================
st.markdown(
    """
    <h2 style='text-align:center;'>🫀 Heart Disease Prediction System</h2>
    <p style='text-align:center; color:gray;'>
    AI-Powered Preliminary Medical Assessment (Prototype)
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ================= INFO CARDS =================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🧬 AI Model")
    st.caption("Deep Neural Network")

with col2:
    st.markdown("### 📊 Dataset")
    st.caption("Clinical Patient Records")

with col3:
    st.markdown("### 🌐 Platform")
    st.caption("Streamlit Web App")

# ================= INTRO =================
st.info(
    "This system predicts the **risk of heart disease** using clinical parameters "
    "and a trained deep learning model. It is intended for **educational and "
    "preliminary assessment purposes only**."
)

# ================= SIDEBAR =================
st.sidebar.markdown("### 🧾 Patient Information")

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

# ================= INPUT =================
user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekg,
                        max_hr, ex_angina, st_depression,
                        slope, vessels, thallium]])

# ================= PREDICTION =================
if st.sidebar.button("🔍 Predict Risk"):

    st.markdown("## 📌 Prediction Outcome")

    prediction = model.predict(user_input)
    probability = prediction[0][0]

    # Visual Probability Bar
    st.progress(float(probability))

    colA, colB = st.columns(2)

    with colA:
        st.metric("Risk Probability", f"{probability:.2f}")

    with colB:
        if probability > 0.5:
            st.error("⚠️ High Risk")
        else:
            st.success("✅ Low Risk")

    # Explanation
    if probability > 0.5:
        st.markdown(
            """
            **Interpretation:**  
            The model currently predicts a **higher risk**, though it may show
            inconsistencies due to its developmental nature. Medical consultation
            is advised.
            """
        )
    else:
        st.markdown(
            """
            **Interpretation:**  
            The model predicts a **lower risk**. This does not replace professional
            medical advice.
            """
        )

# ================= DISCLAIMER =================
st.warning(
    "⚠️ **Disclaimer:** This system is a developing prototype. "
    "Predictions may be inconsistent and should not be used as a final diagnosis."
)

# ================= FOOTER =================
st.markdown(
    """
    <hr>
    <p style='text-align:center; font-size:12px; color:gray;'>
    Developed by Aman Kumar Choudhary<br>
    BCA (Data Science) | AI-Powered Medical Diagnosis System
    </p>
    """,
    unsafe_allow_html=True
)
