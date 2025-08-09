import streamlit as st
import pandas as pd
import joblib

# Load model dan preprocessing
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load('online_food_model.pkl')  # File model tunggal
    return artifacts

artifacts = load_artifacts()
model = artifacts['model']
label_encoders = artifacts['label_encoders']
feature_names = artifacts['feature_names']

# Judul Aplikasi
st.title("🎯 Prediksi Pemesanan Makanan Online")
st.write("Aplikasi ini memprediksi apakah customer akan memesan makanan online atau tidak")

# Input Form
with st.form("input_form"):
    st.header("Data Customer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Umur", min_value=10, max_value=100, value=25)
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
    # Buat DataFrame dengan urutan kolom yang sama seperti saat training
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Marital Status': [marital_status],
        'Occupation': [occupation],
        'Monthly Income': [monthly_income],
        'Educational Qualifications': [education],
        'Family size': [family_size]
    }, columns=feature_names)  # Pastikan urutan kolom sesuai
    
    # Label Encoding untuk input baru
    for col in input_data.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])
    
    # Prediksi
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0]
    
    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.success(f"✅ Customer AKAN memesan (Probabilitas: {proba[1]:.2%})")
    else:
        st.error(f"❌ Customer TIDAK AKAN memesan (Probabilitas: {proba[0]:.2%})")
    
    # Tampilkan feature importance jika tersedia
    if hasattr(model, 'coef_'):
        st.subheader("Faktor yang Mempengaruhi")
        importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Pengaruh': model.coef_[0]
        }).sort_values('Pengaruh', ascending=False)
        st.bar_chart(importance_df.set_index('Fitur'))