import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

model = tf.keras.models.load_model('heart_disease_model.h5')

st.title("❤️Heart-Disease Prediction App")

st.markdown("---")
st.caption("**Built by Aman**")

st.sidebar.header("Enter Patient Information")

age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
bp = st.sidebar.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
ekgr = st.sidebar.slider("EKG Results (0-2)", 0, 2, 1)
max_hr = st.sidebar.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=150)
ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
st_depression = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
slope = st.sidebar.slider("Slope of ST (0-2)", 0, 2, 1)
vessels = st.sidebar.slider("Number of Vessels (0-3)", 0, 3, 0)
thallium = st.sidebar.slider("Thallium Stress Test (1-3)", 1, 3, 2)

user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekgr, max_hr, ex_angina, st_depression, slope, vessels, thallium]])

if st.sidebar.button("Predict Heart Disease"):
    prediction = model.predict(user_input)
    result = prediction[0][0] > 0.5

    st.subheader("🧬 Prediction Result")
    if result:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.info(f"Prediction Probability: {prediction[0][0]:.2f}")

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit")
