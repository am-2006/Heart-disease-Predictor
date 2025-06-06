import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd

# Load the pre-trained model
model = tf.keras.models.load_model('heart_disease_model.h5')

# App title and description
st.title("❤️ Heart-Disease Prediction App")
st.markdown("---")
st.caption("**Built by Aman**")

# Sidebar header for patient information input
st.sidebar.header("Enter Patient Information")

# Input fields for patient information
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=67)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 3)

# Directly using combined Blood Pressure (bp) input
bp = st.sidebar.number_input("Blood Pressure (mm Hg)", min_value=80, max_value=250, value=115)

cholesterol = st.sidebar.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=564)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], index=0)  # Default to 0
ekgr = st.sidebar.slider("EKG Results (0-2)", 0, 2, 2)

max_hr = st.sidebar.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=160)
ex_angina = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)  # Default to No

st_depression = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.6)
slope = st.sidebar.slider("Slope of the ST Segment (0-2)", 0, 2, 2)
vessels = st.sidebar.slider("Number of Major Vessels (0-3)", 0, 3, 0)
thallium = st.sidebar.slider("Thallium Stress Test Result (1-3)", 1, 3, 2)

# Convert input data into a numpy array for prediction
user_input = np.array([[age, sex, cp, bp, cholesterol, fbs, ekgr, max_hr, ex_angina, 
                        st_depression, slope, vessels, thallium]])

# Button to trigger prediction
if st.sidebar.button("Predict Heart Disease"):

    # Predict heart disease using the model
    prediction = model.predict(user_input)
    result = prediction[0][0] > 0.5  # If the output is greater than 0.5, it is considered high risk

    # Show prediction results
    st.subheader("🧬 Prediction Result")
    if result:
        st.error("⚠️ High Risk of Heart Disease")
        st.markdown("### Suggested Actions for High-Risk Patients:")
        st.markdown("""
        - **Seek Medical Consultation:** Schedule an immediate appointment with a cardiologist or primary care provider.
        - **Lifestyle Modifications:**
            - **Diet:** Adopt a heart-healthy diet low in sodium, saturated fat, and processed foods; increase fruits, vegetables, and whole grains.
            - **Physical Activity:** Engage in moderate exercise (e.g., brisk walking) for at least 30 minutes per day, as recommended by your doctor.
            - **Avoid Tobacco & Limit Alcohol:** Quit smoking and limit alcohol consumption.
        - **Regular Monitoring:** 
            - Keep track of your blood pressure, cholesterol, and blood sugar regularly.
            - Consider using home monitoring devices.
        - **Medication Management:** 
            - Follow your prescribed medications faithfully.
            - Discuss any side effects or concerns with your healthcare provider.
        - **Stress Management:** 
            - Practice stress-reduction techniques such as meditation, yoga, or deep-breathing exercises.
        """)
    else:
        st.success("✅ Low Risk of Heart Disease")
        st.markdown("### Maintain Your Heart Health:")
        st.markdown("""
        - Continue following a healthy lifestyle.
        - Attend regular check-ups to monitor your vital signs.
        - Keep an active routine and balanced diet.
        """)

    st.info(f"Prediction Probability: {prediction[0][0]:.2f}")

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit")
