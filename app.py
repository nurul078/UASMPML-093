import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan preprocessing
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_regression_model.pkl')
    preprocessor = joblib.load('preprocessing_pipeline.pkl')
    label_encoders = joblib.load('label_encoders.pkl')  # Simpan ini saat training
    return model, preprocessor, label_encoders

model, preprocessor, label_encoders = load_artifacts()

# Judul Aplikasi
st.title("🎯 Prediksi Usia Customer")
st.write("Aplikasi ini memprediksi usia customer berdasarkan karakteristiknya")

# Input Form
with st.form("input_form"):
    st.header("Data Customer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Prefer not to say"])
        occupation = st.selectbox("Pekerjaan", ["Student", "Employee", "Self Employeed", "House wife"])
    
    with col2:
        monthly_income = st.selectbox("Pendapatan Bulanan", 
                                    ["No Income", "Below Rs.10000", "10001 to 25000", 
                                     "25001 to 50000", "More than 50000"])
        education = st.selectbox("Pendidikan", 
                               ["Graduate", "Post Graduate", "Ph.D", "School", "Uneducated"])
        family_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=10, value=3)
    
    submitted = st.form_submit_button("Prediksi Sekarang!")

if submitted:
    # Buat DataFrame dengan kolom yang sama seperti saat training
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Occupation': [occupation],
        'Monthly Income': [monthly_income],
        'Educational Qualifications': [education],
        'Family size': [family_size]
    })
    
    # Label Encoding untuk input baru
    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])
    
    # Preprocess data
    processed_data = preprocessor.transform(input_data)
    
    # Prediksi
    prediction = model.predict(processed_data)
    
    st.subheader("Hasil Prediksi")
    st.success(f"Prediksi Usia Customer: {int(prediction[0])} tahun")