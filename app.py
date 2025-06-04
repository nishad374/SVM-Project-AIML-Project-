import streamlit as st
import joblib
import numpy as np
import pandas as pd
import tempfile
from fpdf import FPDF
import base64
from datetime import datetime

# 1. Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

# 2. Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load('svm_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# 3. UI Components
st.title("❤️ Heart Disease Prediction")
st.markdown("Using Support Vector Machine (SVM)")

# Credits
st.markdown(
    "<h4 style='color:gray;'>Created by Priyanka, Soni, Manisha, Abhishek & Imran</h4>", 
    unsafe_allow_html=True
)

# NEW: Patient Name Input
patient_name = st.text_input("Patient Name", "John Doe")

# Input Sections
st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 50)
    sex = st.radio("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", 
                      ["Typical Angina", "Atypical Angina", 
                       "Non-anginal Pain", "Asymptomatic"])

with col2:
    trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 120)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

st.header("Exercise & ECG Results")
col3, col4 = st.columns(2)

with col3:
    thalach = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

with col4:
    restecg = st.selectbox("Resting ECG", 
                           ["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                         ["Upsloping", "Flat", "Downsloping"])

st.header("Cardiac Measurements")
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", 
                    ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert inputs to model format
sex_val = 1 if sex == "Male" else 0
fbs_val = 1 if fbs == "Yes" else 0
exang_val = 1 if exang == "Yes" else 0

cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}

cp_val = cp_map[cp]
restecg_val = restecg_map[restecg]
slope_val = slope_map[slope]
thal_val = thal_map[thal]

# Create input as DataFrame
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

input_data = pd.DataFrame([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                            restecg_val, thalach, exang_val, oldpeak,
                            slope_val, ca, thal_val]], columns=columns)

# Prediction
if st.button("Predict Heart Disease Risk"):
    processed_data = preprocessor.transform(input_data)
    prediction_proba = model.predict_proba(processed_data)
    probability = prediction_proba[0][1]

    # Define risk level
    risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.4 else "Low"
    color = "red" if risk_level == "High" else "orange" if risk_level == "Moderate" else "green"

    # Display results
    st.subheader("Prediction Results")
    st.metric("Probability of Heart Disease", f"{probability:.1%}")

    st.markdown(f"**Risk Level**: <span style='color:{color};font-weight:bold'>{risk_level}</span>", 
                unsafe_allow_html=True)

    st.progress(int(probability * 100))

    # Interpretation
    if risk_level == "High":
        st.error("High risk detected! Consult a cardiologist immediately.")
    elif risk_level == "Moderate":
        st.warning("Moderate risk detected. Consider further medical evaluation.")
    else:
        st.success("Low risk detected. Maintain a healthy lifestyle.")

    # PDF Generation
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Report header with patient name
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Heart Disease Risk Assessment Report", ln=1, align='C')
    pdf.set_font("Arial", size=12)
    
    # NEW: Add patient name to report
    pdf.cell(200, 10, txt=f"Patient Name: {patient_name}", ln=1)
    pdf.cell(200, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(10)
    
    # Patient information section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Patient Information", ln=1)
    pdf.set_font("Arial", size=12)
    
    # Create patient info table
    patient_data = [
        # NEW: Patient name in report
        ["Patient Name", patient_name],
        ["Age", str(age)],
        ["Sex", sex],
        ["Chest Pain Type", cp],
        ["Resting BP", f"{trestbps} mm Hg"],
        ["Cholesterol", f"{chol} mg/dl"],
        ["Fasting Blood Sugar > 120", fbs],
        ["Resting ECG", restecg],
        ["Max Heart Rate", str(thalach)],
        ["Exercise Induced Angina", exang],
        ["ST Depression", str(oldpeak)],
        ["ST Segment Slope", slope],
        ["Major Vessels", str(ca)],
        ["Thalassemia", thal]
    ]
    
    for row in patient_data:
        pdf.cell(95, 8, txt=row[0], border=0)
        pdf.cell(95, 8, txt=row[1], border=0, ln=1)
    
    # Prediction results section
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Prediction Results", ln=1)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(95, 8, txt="Heart Disease Probability:", border=0)
    pdf.cell(95, 8, txt=f"{probability:.1%}", border=0, ln=1)
    
    pdf.cell(95, 8, txt="Risk Level:", border=0)
    pdf.cell(95, 8, txt=risk_level, border=0, ln=1)
    
    # Interpretation note
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 8, txt="Note: This report is generated by an automated system. "
                            "Consult a healthcare professional for medical advice.")
    
    # Save PDF to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        pdf_bytes = tmpfile.read()
    
    # Create download button
    st.download_button(
        label="📥 Download Report as PDF",
        data=pdf_bytes,
        file_name=f"heart_disease_report_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )