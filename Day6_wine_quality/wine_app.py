import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("wine-quality-model.pkl","rb") as f: 
	model,feature_names = pickle.load(f)

st.title("🍷 Wine Quality Predictor")
st.markdown("Enter the chemical properties below:")

#slidebar inputs 

input_data = {}

for feature in feature_names: 

	input_data[feature] = st.sidebar.slider(

		label = feature.capitalize().replace("_"," "),
		min_value = float(0),
		max_value = float(15),
		value = float(5),
		step=0.1




	)
input_df = pd.DataFrame([input_data])

if st.button("predict"): 
	prediction = model.predict(input_df)[0]
	st.success(f"Predicted wine quality {prediction}")
	st.info("0: Worst Quality, 10 : Best Qaulity ")