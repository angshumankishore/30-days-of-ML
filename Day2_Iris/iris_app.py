import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")
st.title("🌸 Iris Flower Classifier")
st.write("Enter flower measurements to classify the species.")

sepal_length = st.slider("Sepal width (cm)",4.0,8.0,5.1)
sepal_width = st.slider("Sepal width (cm)",2.0,5.0,3.1)
petal_length = st.slider("Petal width (cm)",1.0,7.0,1.4)
petal_width= st.slider("Petal width (cm)",0.1,2.5,0.2)

features = np.array([[sepal_length,sepal_width,petal_length,petal_width]])

if(st.button("Predict")): 
    prediction = model.predict(features)[0]
    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"🌼 Predicted Species: **{species[prediction]}**")

