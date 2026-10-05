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
attendance = st.number_input(
    "Student attendance (%)", min_value=0, max_value=100, value=85, step=1
)
projects = st.number_input(
    "Number of projects completed", min_value=0, max_value=20, value=3, step=1
)

if st.button("Predict Placement", type="primary"):
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train_model.py` first.")
        st.stop()

    model = joblib.load(MODEL_PATH)
    student = pd.DataFrame(
        [[cgpa, iq, attendance, projects]],
        columns=["cgpa", "iq", "attendance", "projects"],
    )
    prediction = model.predict(student)[0]
    placement_probability = model.predict_proba(student)[0][1]

    if prediction == 1:
        st.success("High likelihood of getting placed")
    else:
        st.error("Low likelihood of getting placed")
    st.metric("Estimated placement likelihood", f"{placement_probability:.0%}")