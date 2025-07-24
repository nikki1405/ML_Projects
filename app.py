import streamlit as st
import numpy as np
import joblib

# Load saved model
model = joblib.load("breast_cancer_model.pkl")

st.set_page_config(page_title="Breast Cancer Predictor")
st.title("🧬 Breast Cancer Prediction App")
st.markdown("Predict whether a tumor is Malignant or Benign based on input features.")

# Define 10 features (you can expand based on your model)
feature_names = [
    "mean radius", "mean texture", "mean perimeter",
    "mean area", "mean smoothness", "mean compactness",
    "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension"
]

input_data = []

st.sidebar.header("Input Features")

for feature in feature_names:
    val = st.sidebar.slider(feature, 0.0, 100.0, 10.0)
    input_data.append(val)

# Predict
if st.button("Predict"):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    result = "Malignant" if prediction == 0 else "Benign"
    st.success(f"🎯 Prediction: The tumor is **{result}**.")
