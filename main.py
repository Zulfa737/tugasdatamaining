import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

# Load model & tools
@st.cache_resource
def load_model_objects():
    try:
        # Pastikan nama file sesuai dengan yang disimpan di tahap training
        model_nb = joblib.load("model_churn_nb.pkl")
        model_svm = joblib.load("model_churn_svm.pkl")
        model_ensemble = joblib.load("model_churn_ensemble.pkl")
        scaler = joblib.load("scaler_churn.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model_nb, model_svm, model_ensemble, scaler, feature_names
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file .pkl tersedia. Error: {e}")
        return None, None, None, None, None

model_nb, model_svm, model_ensemble, scaler, feature_names = load_model_objects()

# Fungsi untuk memproses input user agar sesuai dengan format training
def preprocess_input(data, scaler, feature_names):
    # Ubah dictionary ke DataFrame
    df = pd.DataFrame([data])
    
    # One-Hot Encoding (sama seperti saat training)
    # Kita perlu mendefinisikan kolom kategori yang sama persis dengan data training awal
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    # Lakukan get_dummies
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    
    # PENTING: Menyelaraskan kolom dengan data training
    # Jika user memilih 'DSL', kolom 'InternetService_Fiber optic' tidak akan terbentuk otomatis.
    # Kita harus memaksanya ada (diisi 0) agar model tidak error.
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Scaling data numerik (Tenure, MonthlyCharges, TotalCharges)
    # Lokasi kolom numerik harus sama dengan saat fit_transform di training
    # Kita update nilai di dataframe yang sudah di-reindex
    
    # Ambil kolom numerik dari df asli untuk di-scale (karena reindex mungkin mengacak urutan jika tidak hati-hati)
    # Namun, scaler sklearn bekerja berdasarkan posisi kolom atau nama fitur jika dataframe.
    # Untuk aman, kita scale dataframe hasil reindex.
    
    df_scaled = scaler.transform(df_encoded)
    
    return df_scaled

# Confidence badge helper
def get_confidence_badge(prob):
    if prob > 80:
        return "🔴 Tinggi (Sangat Mungkin Churn)", "error"
    elif prob > 50:
        return "🟡 Sedang (Berpotensi Churn)", "warning"
    else:
        return "🟢 Aman (Kemungkinan Kecil Churn)", "success"

# === UI UTAMA ===

st.title("📉 Prediksi Churn Pelanggan Telco")
st.markdown("""
Aplikasi ini memprediksi apakah pelanggan akan berhenti berlangganan (**Churn**) 
berdasarkan data profil dan penggunaan mereka.
""")

if model_ensemble is None:
    st.warning("⚠️ File model (.pkl) belum di-upload atau tidak ditemukan.")
else:
    # Form Input
    st.sidebar.header("📝 Input Data Pelanggan")
    
    with st.sidebar.form("user_input_form"):
        # Data Numerik
        st.subheader("Data Layanan")
        tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Biaya Bulanan (USD)", min_value=0.0, value=50.0)
        total_charges = st.number_input("Total Biaya (USD)", min_value=0.0, value=600.0)
        
        st.subheader("Data Pribadi")
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Memiliki Pasangan?", ["Yes", "No"])
        dependents = st.selectbox("Memiliki Tanggungan?", ["Yes", "No"])
        senior_citizen = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: "Ya" if x==1 else "Tidak")
        
        st.subheader("Detail Layanan")
        contract = st.selectbox("Jenis Kontrak", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Layanan Internet", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        # Fitur tambahan (Default No/Yes)
        with st.expander("Fitur Tambahan (Klik untuk buka)"):
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

        submit_btn = st.form_submit_button("🔍 Prediksi Churn")

    # Logic Prediksi
    if submit_btn:
        # Kumpulkan data dalam dictionary
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'SeniorCitizen': senior_citizen,
            'gender': gender,
            'Partner': partner,
            'Dependents': dependents,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method
        }
        
        with st.spinner('Memproses data...'):
            try:
                # Preprocessing
                X_input = preprocess_input(input_data, scaler, feature_names)
                
                # Prediksi
                pred_ensemble = model_ensemble.predict(X_input)[0]
                prob_ensemble = model_ensemble.predict_proba(X_input)[0]
                
                # Tampilan Hasil
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Hasil Prediksi (Ensemble Model)")
                    
                    # Karena 1 = Churn (Yes), 0 = No Churn (No)
                    churn_prob = prob_ensemble[1] * 100 
                    conf_text, conf_type = get_confidence_badge(churn_prob)
                    
                    if pred_ensemble == 1:
                        st.error(f"### 🚨 Prediksi: PELANGGAN AKAN CHURN (Berhenti)")
                        st.write(f"Pelanggan ini memiliki risiko tinggi untuk berhenti berlangganan.")
                    else:
                        st.success(f"### ✅ Prediksi: PELANGGAN SETIA (Tidak Churn)")
                        st.write(f"Pelanggan ini diprediksi akan tetap berlangganan.")
                    
                    st.info(f"Tingkat Risiko Churn: **{conf_text}** ({churn_prob:.1f}%)")

                with col2:
                    st.write("📊 **Probabilitas Detail**")
                    st.metric("Akan Stay (No)", f"{prob_ensemble[0]*100:.1f}%")
                    st.metric("Akan Churn (Yes)", f"{prob_ensemble[1]*100:.1f}%")

                # Perbandingan Model Lain (Opsional)
                with st.expander("Lihat Pendapat Model Lain"):
                    pred_nb = model_nb.predict(X_input)[0]
                    pred_svm = model_svm.predict(X_input)[0]
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption("Gaussian Naive Bayes")
                        st.write("⛔ Churn" if pred_nb == 1 else "✅ Stay")
                    with c2:
                        st.caption("Linear SVM")
                        st.write("⛔ Churn" if pred_svm == 1 else "✅ Stay")

            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan data: {e}")
                st.write("Detail Error:", e)
