import streamlit as st
import pandas as pd
import numpy as np
import joblib


model = joblib.load("diabetes_model.pkl")

st.title("Diabetes Prediction Model")

#user inputs 
preg = st.number_input('Pregnancies', 0, 20,10)
glucose = st.number_input('Glucose', 0, 200,100)
bp = st.number_input('Blood Pressure', 0, 150,75)
skin = st.number_input('Skin Thickness', 0, 100,50)
insulin = st.number_input('Insulin', 0, 900,400)
bmi = st.number_input('BMI', 0.0, 70.0,35.0)
dpf = st.number_input('Diabetes Pedigree Function', 0.0, 3.0,1.5)
age = st.number_input('Age', 1, 120,60)


if st.button('Predict'): 
	input_data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
	prediction = model.predict(input_data)
	st.success(f'The Patient is {"Diabetic " if prediction[0] else " Not Diabetic"}')