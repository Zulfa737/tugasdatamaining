import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Churn Pelanggan",
    page_icon="📉",
    layout="centered"
)

# --- 2. Fungsi Load Model (Membaca file .pkl) ---
@st.cache_resource
def load_model_objects():
    # Daftar file yang wajib ada
    required_files = [
        "model_churn_ensemble.pkl",
        "model_churn_nb.pkl",
        "model_churn_svm.pkl",
        "scaler_churn.pkl",
        "feature_names.pkl"
    ]
    
    # Cek keberadaan file
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        st.error(f"⚠ File model berikut hilang: {', '.join(missing_files)}")
        st.info("Pastikan Anda meng-upload semua file .pkl dari Google Colab ke folder GitHub yang sama dengan main.py")
        return None, None, None, None, None

    try:
        # Load file menggunakan joblib
        model_ensemble = joblib.load("model_churn_ensemble.pkl")
        model_nb = joblib.load("model_churn_nb.pkl")
        model_svm = joblib.load("model_churn_svm.pkl")
        scaler = joblib.load("scaler_churn.pkl")
        feature_names = joblib.load("feature_names.pkl")
        
        return model_ensemble, model_nb, model_svm, scaler, feature_names
        
    except Exception as e:
        st.error(f"Gagal membaca file model. Error: {e}")
        return None, None, None, None, None

# Panggil fungsi load
model_ensemble, model_nb, model_svm, scaler, feature_names = load_model_objects()

# --- 3. Fungsi Preprocessing Input ---
def preprocess_input(data, scaler, feature_names):
    # 1. Ubah dictionary input ke DataFrame
    df = pd.DataFrame([data])
    
    # 2. One-Hot Encoding (Sama seperti training)
    df_encoded = pd.get_dummies(df)
    
    # 3. Penyelarasan Kolom (CRITICAL STEP)
    # Agar urutan dan jumlah kolom sama persis dengan feature_names.pkl
    # Jika ada kolom kurang (misal user tidak pilih 'Fiber Optic'), akan diisi 0.
    df_final = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # 4. Scaling (Normalisasi angka)
    df_scaled = scaler.transform(df_final)
    
    return df_scaled

# --- 4. Helper UI (Badge Warna) ---
def get_confidence_badge(prob):
    if prob > 75:
        return "🔴 Tinggi", "error"
    elif prob > 50:
        return "🟡 Sedang", "warning"
    else:
        return "🟢 Rendah", "success"

# --- 5. UI UTAMA ---
st.title("📉 Analisis Churn Pelanggan")
st.markdown("### Ensemble Model (Loaded from Colab)")

# Cek apakah model berhasil di-load
if model_ensemble is None:
    st.warning("Menunggu file model (.pkl) diupload...")
else:
    st.subheader("✍ Masukkan Data Profil")

    # --- Pilihan Contoh Profil ---
    example_profiles = {
        "Contoh 1: Pelanggan Baru (Risiko Tinggi)": {
            'tenure': 1, 'MonthlyCharges': 70.0, 'TotalCharges': 70.0, 
            'Contract': 'Month-to-month', 'InternetService': 'Fiber optic', 
            'PaymentMethod': 'Electronic check', 'OnlineSecurity': 'No', 'TechSupport': 'No'
        },
        "Contoh 2: Pelanggan Lama (Setia)": {
            'tenure': 60, 'MonthlyCharges': 20.0, 'TotalCharges': 1200.0, 
            'Contract': 'Two year', 'InternetService': 'No', 
            'PaymentMethod': 'Mailed check', 'OnlineSecurity': 'No internet service', 'TechSupport': 'No internet service'
        },
        "Contoh 3: Pelanggan Ragu-ragu": {
            'tenure': 12, 'MonthlyCharges': 65.0, 'TotalCharges': 780.0, 
            'Contract': 'One year', 'InternetService': 'DSL', 
            'PaymentMethod': 'Bank transfer (automatic)', 'OnlineSecurity': 'No', 'TechSupport': 'Yes'
        }
    }

    selected_example = st.selectbox(
        "Pilih contoh profil:",
        ["-- Ketik manual --"] + list(example_profiles.keys())
    )

    defaults = example_profiles[selected_example] if selected_example != "-- Ketik manual --" else None

    # --- Input Form ---
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input("Lama Langganan (Bulan)", 0, 100, value=defaults['tenure'] if defaults else 12)
            contract = st.selectbox("Kontrak", ['Month-to-month', 'One year', 'Two year'], index=['Month-to-month', 'One year', 'Two year'].index(defaults['Contract']) if defaults else 0)
            monthly = st.number_input("Biaya Bulanan ($)", 0.0, 500.0, value=defaults['MonthlyCharges'] if defaults else 50.0)
            tech = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(defaults['TechSupport']) if defaults else 0)
        with c2:
            internet = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'], index=['DSL', 'Fiber optic', 'No'].index(defaults['InternetService']) if defaults else 0)
            payment = st.selectbox("Pembayaran", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(defaults['PaymentMethod']) if defaults else 0)
            total = st.number_input("Total Biaya ($)", 0.0, 10000.0, value=defaults['TotalCharges'] if defaults else 100.0)
            security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(defaults['OnlineSecurity']) if defaults else 0)

    # --- Tombol Aksi ---
    col1, col2, col3 = st.columns(3)
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary")
    with col2:
        show_comparison = st.checkbox("Bandingkan model", value=True)
    with col3:
        show_details = st.checkbox("Detail preprocessing", value=False)

    if predict_btn:
        with st.spinner('Menganalisis data...'):
            try:
                # 1. Siapkan Data Input Dictionary
                # Kita set default value untuk fitur lain (gender, partner, dll) 
                # karena fitur tersebut tidak ada di form input tapi dibutuhkan model.
                input_data = {
                    'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total,
                    'Contract': contract, 'InternetService': internet, 'PaymentMethod': payment,
                    'OnlineSecurity': security, 'TechSupport': tech,
                    # Default values (Asumsi)
                    'gender': 'Male', 'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes',
                    'MultipleLines': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
                    'StreamingTV': 'No', 'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'SeniorCitizen': 0
                }

                # 2. Preprocess (Encoding & Scaling)
                input_scaled = preprocess_input(input_data, scaler, feature_names)

                # 3. Prediksi (Ensemble)
                pred_ensemble = model_ensemble.predict(input_scaled)[0]
                prob_ensemble = model_ensemble.predict_proba(input_scaled)[0]

                # 4. Prediksi Model Lain (Untuk Perbandingan)
                prob_nb = model_nb.predict_proba(input_scaled)[0]
                pred_nb = model_nb.predict(input_scaled)[0]
                
                prob_svm = model_svm.predict_proba(input_scaled)[0]
                pred_svm = model_svm.predict(input_scaled)[0]

                # ======== Hasil Ensemble ========
                st.subheader("🎯 Hasil Analisis (Ensemble)")

                # Class 1 = Churn (Positif), Class 0 = Stay (Negatif)
                churn_prob = prob_ensemble[1] * 100
                conf_text, conf_type = get_confidence_badge(churn_prob)

                if pred_ensemble == 1:
                    st.error("### ❌ Prediksi: AKAN CHURN (Berhenti)")
                    st.write("Pelanggan ini memiliki kecenderungan tinggi untuk berhenti berlangganan.")
                else:
                    st.success("### ✅ Prediksi: SETIA (Stay)")
                    st.write("Pelanggan ini diprediksi akan tetap menggunakan layanan.")

                st.info(f"Tingkat Risiko Churn: {conf_text} ({churn_prob:.1f}%)")

                # Metrics Probabilitas
                st.write("📊 Probabilitas:")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Akan Stay", f"{prob_ensemble[0]*100:.1f}%")
                with m2:
                    st.metric("Akan Churn", f"{prob_ensemble[1]*100:.1f}%")

                # ======== Bandingkan Model ========
                if show_comparison:
                    st.divider()
                    st.subheader("📌 Perbandingan Model")

                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.metric("Naive Bayes", 
                                  "CHURN" if pred_nb == 1 else "STAY",
                                  f"{prob_nb[1]*100:.1f}% Risk")
                    with k2:
                        st.metric("Linear SVM", 
                                  "CHURN" if pred_svm == 1 else "STAY",
                                  f"{prob_svm[1]*100:.1f}% Risk")
                    with k3:
                        st.metric("Ensemble", 
                                  "CHURN" if pred_ensemble == 1 else "STAY",
                                  f"{prob_ensemble[1]*100:.1f}% Risk")

                # ======== Detail Preprocessing ========
                if show_details:
                    st.divider()
                    st.subheader("🔎 Detail Preprocessing")
                    st.text("Input Mentah (User):")
                    st.write(input_data)
                    st.text("Input Scaled (Masuk ke Model):")
                    # Tampilkan dataframe hasil scaling untuk debugging
                    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names).head())

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
