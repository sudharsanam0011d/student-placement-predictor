"""Streamlit interface for predicting student placement likelihood."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "model.pkl"

st.set_page_config(page_title="Student Placement Predictor", page_icon="🎓")
st.title("🎓 Student Placement Predictor")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
iq = st.number_input("IQ", min_value=50, max_value=200, value=110, step=1)

if st.button("Predict Placement", type="primary"):
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    student = pd.DataFrame([[cgpa, iq]], columns=["cgpa", "iq"])
    prediction = model.predict(student)[0]

    if prediction == 1:
        st.success("🎉 High likelihood of getting placed!")
    else:
        st.error("⚠️ Low likelihood of getting placed.")