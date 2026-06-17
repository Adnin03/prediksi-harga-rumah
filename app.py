import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Model dan Kolom
model = joblib.load('model_rumah_uika.pkl')
daftar_kolom = joblib.load('daftar_kolom.pkl')

# 2. Setup Tampilan Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah Jakarta Barat", layout="centered")
st.title("🏠 AI Estimasi Harga Rumah")
st.write("Aplikasi ini memprediksi harga rumah di Jakarta Barat berdasarkan data historis.")

# 3. Form Input User
with st.sidebar:
    st.header("Spesifikasi Rumah")
    luas_tanah = st.number_input("Luas Tanah (m2)", min_value=1, value=150)
    luas_bangunan = st.number_input("Luas Bangunan (m2)", min_value=1, value=120)
    kamar_tidur = st.slider("Jumlah Kamar Tidur", 1, 10, 3)
    kamar_mandi = st.slider("Jumlah Kamar Mandi", 1, 10, 2)
    garasi = st.slider("Kapasitas Garasi (Mobil)", 0, 10, 1)
    
    # Ambil daftar kecamatan unik dari kolom One-Hot
    list_kecamatan = [col.replace('Kecamatan_', '') for col in daftar_kolom if 'Kecamatan_' in col]
    kecamatan_pilihan = st.selectbox("Pilih Kecamatan", sorted(list_kecamatan))

# 4. Logika Prediksi
if st.button("Estimasi Harga Sekarang"):
    # Buat DataFrame untuk input
    input_df = pd.DataFrame(columns=daftar_kolom)
    input_df.loc[0] = 0 # Isi awal dengan 0
    
    # Isi data numerik
    input_df['Luas Tanah'] = luas_tanah
    input_df['Luas Bangunan'] = luas_bangunan
    input_df['Kamar Tidur'] = kamar_tidur
    input_df['Kamar Mandi'] = kamar_mandi
    input_df['Garasi'] = garasi
    
    # Isi data kategori (One-Hot)
    kolom_kec = f"Kecamatan_{kecamatan_pilihan}"
    if kolom_kec in daftar_kolom:
        input_df[kolom_kec] = 1
        
    # Prediksi
    pred_log = model.predict(input_df)
    pred_asli = np.expm1(pred_log)[0]
    
    # Tampilkan Hasil
    st.success(f"### Estimasi Harga: Rp {pred_asli:,.0f}")
    st.info(f"Rumah di {kecamatan_pilihan} dengan LT {luas_tanah}m2 dan LB {luas_bangunan}m2.")