import streamlit as st
import numpy as np
import tensorflow as tf

# --- PHASE 3: BUILDING THE STREAMLIT UI ---

# 1. Configure the Page
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide"
)

# 2. Custom Styling (Injecting CSS for Phase 3 "Clinical Cards")
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e9ecef;
    }
    .status-high { color: #d62828; font-weight: bold; font-size: 1.2rem; }
    .status-low { color: #2a9d8f; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# 3. Load the Model (Required for Phase 4)
@st.cache_resource
def load_trained_model():
    """Loads the .h5 model file created in Phase 2."""
    try:
        return tf.keras.models.load_model("heart_disease_model.h5")
    except Exception as e:
        return None

model = load_trained_model()

# 4. Create the Sidebar (Phase 3 Inputs)
st.sidebar.header("📋 Patient Clinical Inputs")
st.sidebar.markdown("Enter medical markers below:")

age = st.sidebar.number_input("Age", 20, 100, 55)
sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol", 100, 600, 240)
fbs = st.sidebar.radio("Fasting Blood Sugar > 120 mg/dl", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.radio("Exercise Induced Angina", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
oldpeak = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Peak Exercise ST Slope", [0, 1, 2])
ca = st.sidebar.slider("Number of Major Vessels", 0, 3, 0)
thal = st.sidebar.selectbox("Thallium Test Result", [1, 2, 3])

# --- PHASE 4: INTEGRATION & PREDICTION LOGIC ---

st.title("Cardiovascular Risk Analysis")
st.markdown("Use this AI-assisted tool for preliminary patient screening and clinical triage.")

if st.sidebar.button("Run Diagnostic Assessment"):
    if model is not None:
        # 1. Array Conversion
        user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, 
                               thalach, exang, oldpeak, slope, ca, thal]])
        
        # 2. Prediction
        prediction = model.predict(user_input)
        probability = float(prediction[0][0])
        
        # 3. Thresholding & Results
        st.divider()
        col_res1, col_res2 = st.columns([2, 1])
        
        with col_res1:
            if probability > 0.5:
                st.error(f"### Assessment: HIGH RISK ({probability:.1%})")
                st.write("Clinical indicators show high correlation with heart disease patterns.")
            else:
                st.success(f"### Assessment: LOW RISK ({probability:.1%})")
                st.write("Clinical markers suggest a lower probability of heart disease.")
        
        with col_res2:
            st.metric("Risk Probability", f"{probability:.1%}")
            st.progress(probability)

        # --- PHASE 5: ADDING CLINICAL CONTEXT ---

        st.subheader("📋 Clinical Guidance")
        
        # 1. Dynamic Content
        if probability > 0.5:
            st.markdown(f"""
            <div class='card'>
                <p class='status-high'>⚠️ Urgent Recommendations:</p>
                <ul>
                    <li>Immediate referral to a cardiologist for diagnostic correlation.</li>
                    <li>Evaluate for <strong>Stress ECG</strong> or <strong>Echocardiogram</strong>.</li>
                    <li>Strict monitoring of lipid profile and blood pressure.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card'>
                <p class='status-low'>✅ Preventive Recommendations:</p>
                <ul>
                    <li>Maintain 150 minutes of moderate aerobic activity weekly.</li>
                    <li>Continue heart-healthy diet (Mediterranean/DASH).</li>
                    <li>Schedule annual cardiovascular wellness screenings.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # 2. External Links
        st.link_button("🏥 Find Nearby Cardiac Care Centers", 
                       "https://www.google.com/maps/search/cardiology+hospital+near+me")

else:
    st.info("👈 Enter patient clinical markers in the sidebar and click 'Run Diagnostic Assessment'.")

# Footer Disclaimer
st.divider()
st.caption("Disclaimer: This tool is for educational/screening purposes and does not replace professional medical diagnosis.")
