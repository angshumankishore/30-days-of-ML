import streamlit as st
import numpy as np
import joblib

model = joblib.load('boston_model.pkl')
scaler = joblib.load('boston_scaler.pkl')
features = joblib.load('features_name.pkl')

st.title("🏠 Boston Housing Price Estimator")
st.write("Fill in the values to predict the estimated house price.")

user_input = []
for feature in features: 
	val = st.number_input(f'{feature}',value=0.0,step=0.1)
	user_input.append(val)

if st.button("Predict"): 
	sacled_input = scaler.transform([user_input])
	prediction = model.predict(sacled_input)
	st.success(f"Estimated House Price is :  ${prediction[0]*1000:.2f}")