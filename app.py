import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model_heart.pkl')

st.title("Prediksi Penyakit Jantung")

# Input data user
umur = st.number_input('Umur', 0)
cp = st.number_input('Chest Pain Type (cp)', 0)
trestbps = st.number_input('Resting Blood Pressure', 0)
chol = st.number_input('Cholesterol', 0)
fbs = st.number_input('Fasting Blood Sugar', 0)
restecg = st.number_input('Resting ECG', 0)
thalach = st.number_input('Max Heart Rate Achieved', 0)
exang = st.number_input('Exercise Induced Angina', 0)
oldpeak = st.number_input('Oldpeak', 0.0)
slope = st.number_input('Slope', 0)
ca = st.number_input('Number of major vessels (ca)', 0)
thal = st.number_input('Thalassemia', 0)

# Prediksi saat tombol diklik
if st.button('Prediksi'):
    features = np.array([umur, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(features)

    result = "POTENSI PENYAKIT JANTUNG" if prediction[0] == 1 else "TIDAK TERDETEKSI"
    st.success(f'Hasil prediksi: {result}')
