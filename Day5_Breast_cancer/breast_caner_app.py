import streamlit as st
import numpy as np
import joblib

model = joblib.load("brest_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🔬 Breast Cancer Prediction")

st.write("Enter the following features:")


mean_radius = st.number_input("Mean Radius: ")
mean_texture = st.number_input("Mean Texture: ")
mean_perimeter = st.number_input("Mean Perimeter: ")
mean_area = st.number_input("Mean area: ")
mean_smoothness = st.number_input("Mean Smoothness: ")

if(st.button("Predict")): 

	input_data = np.array([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness]])
	input_scaled = scaler.transform(input_data)
	prediction = model.predict(input_scaled)[0]


	if prediction == 0: 
		st.error("Prediction: Malignant Tumor ❗")
	else: 
		st.success("Prediction: Benign Tumor ✅")
