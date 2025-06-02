# app_fake_news.py

import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title("📰 Fake News Detector")

st.markdown("Enter a news article and the app will predict whether it's real or fake.")

# User input
user_input = st.text_area("Enter News Article Text", height=300)

if st.button("Predict"):
    vec_input = vectorizer.transform([user_input])
    prediction = model.predict(vec_input)[0]
    label = "✅ Real News" if prediction == 1 else "❌ Fake News"
    st.success(f"Prediction: {label}")

