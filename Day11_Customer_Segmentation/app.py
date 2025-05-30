# app_manual_input.py

import streamlit as st
import joblib
import numpy as np

# Load model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("kmeans_scaler.pkl")

st.title("🧑‍🤝‍🧑 Customer Segmentation Predictor")

st.markdown("Enter your values below and see which customer group you belong to!")

# Manual input
income = st.slider("Annual Income (k$)", min_value=10, max_value=150, step=1, value=60)
spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, step=1, value=50)

if st.button("Predict Cluster"):
    # Prepare input
    user_input = np.array([[income, spending]])
    user_input_scaled = scaler.transform(user_input)

    # Predict cluster
    cluster = kmeans.predict(user_input_scaled)[0]
    st.success(f"🎯 You belong to Cluster #{cluster}")
