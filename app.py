import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("breast_cancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Predictor")
st.title("🧬 Breast Cancer Prediction App")
st.markdown("Predict whether a tumor is **Malignant** or **Benign** based on input features.")

# Full list of 30 features
feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

input_data = []

st.sidebar.header("🔧 Input Tumor Features")

# Collect values from user for all 30 features
for feature in feature_names:
    val = st.sidebar.slider(label=feature, min_value=0.0, max_value=100.0, value=10.0, step=0.1)
    input_data.append(val)

# Predict
if st.button("🧪 Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "Malignant" if prediction == 0 else "Benign"

    st.success(f"🎯 Prediction Result: The tumor is **{result}**.")
