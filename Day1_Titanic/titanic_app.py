import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("titanic_model.pkl")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below:")

pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)",[1,2,3])
sex = st.radio("Sex",['male','female'])
age = st.slider("Age",0,100,30)
sibsp = st.number_input("Siblings/Spouses aboard",min_value=0,step=1)
parch = st.number_input("Parents/Children aboard",min_value=0,step=1)
fare = st.number_input("Fare",min_value=0.0,step=0.1)
embarked = st.selectbox("Port of Embarkment ",['S','Q','C'])

sex_encoded = 0 if sex == 'male' else 1
embarked_map = {'S':0 , 'C':1,'Q':2}
embarked = embarked_map[embarked]

input_data = pd.DataFrame({
    'Pclass':[pclass],
    'Sex':[sex_encoded],
    'Age':[age],
    'SibSp':[sibsp],
    'Parch':[parch],
    'Fare':[fare],
    'Embarked':[embarked]
    
})

input_data['PassengerId'] = [999]

# Ensure column order matches training
expected_columns = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
input_data = input_data[expected_columns]

if st.button("Predict"): 
    prediction = model.predict(input_data)[0]
    st.success("🎉 Survived!" if prediction == 1 else "💀 Did not survive.")
