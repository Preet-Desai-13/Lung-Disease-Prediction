import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Lung Disease Predictor", layout="centered")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("model/lung_model.pkl", "rb"))
feature_names = pickle.load(open("model/features.pkl", "rb"))

# -------------------- TITLE --------------------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🫁 Lung Disease Prediction System</h1>", unsafe_allow_html=True)
st.write("### Enter Patient Details")

# -------------------- FUNCTION --------------------
def convert(value):
    return 1 if value == "Yes" else 0

# -------------------- INPUT UI --------------------
st.write("Please fill all details correctly for an accurate prediction.")

input_values = {}

# Grouping features for better UI
cols = st.columns(2)
for i, feat in enumerate(feature_names):
    if feat in ["index", "Patient Id"]:
        input_values[feat] = 0  # Dummy value for IDs
        continue
    
    with cols[i % 2]:
        if feat == "Age":
            input_values[feat] = st.slider("Age", 0, 100, 30)
        elif feat == "Gender":
            val = st.selectbox("Gender", ["Male", "Female"])
            input_values[feat] = 1 if val == "Male" else 2 # Typically 1=Male, 2=Female in this dataset
        else:
            # Most features in this dataset are 1-8 rating
            input_values[feat] = st.slider(f"{feat}", 1, 8, 4)

# -------------------- INPUT ARRAY --------------------
input_data = np.array([[input_values[feat] for feat in feature_names]])

# -------------------- PDF FUNCTION --------------------
def generate_pdf(result, probability):
    file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Lung Disease Prediction Report", styles['Title']))
    content.append(Paragraph(f"Result: {result}", styles['Normal']))
    content.append(Paragraph(f"Risk Score: {probability:.2f}%", styles['Normal']))

    doc.build(content)
    return file_path

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Disease"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1] * 100

    st.write("### 📊 Prediction Result")

    if prediction[0] == 1:
        result_text = "High Risk of Lung Cancer"
        st.error(f"⚠ {result_text}\n\nRisk Score: {probability:.2f}%")
    else:
        result_text = "Low Risk"
        st.success(f"✅ {result_text}\n\nRisk Score: {probability:.2f}%")

    # Progress bar
    st.progress(int(probability))

    # Recommendation
    st.write("### 🩺 Recommendation")

    if probability > 70:
        st.warning("Consult a doctor immediately and go for medical tests.")
    elif probability > 40:
        st.info("Moderate risk. Maintain healthy lifestyle and consider checkup.")
    else:
        st.success("You are relatively safe. Maintain good habits.")

    # -------------------- PDF DOWNLOAD --------------------
    st.write("### 📄 Download Report")

    pdf_file = generate_pdf(result_text, probability)

    with open(pdf_file, "rb") as f:
        st.download_button("Download PDF Report", f, file_name="Lung_Report.pdf")

    # -------------------- FEATURE IMPORTANCE --------------------
    if st.checkbox("Show Feature Importance"):

        try:
            # Importance from model features

            importance = model.feature_importances_

            df_imp = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.bar_chart(df_imp.set_index("Feature"))

        except:
            st.warning("Feature importance not available for this model.")