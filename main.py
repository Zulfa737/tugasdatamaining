import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Analisis Churn Pelanggan",
    page_icon="📉",
    layout="centered"
)

# --- FUNGSI TRAINING MODEL OTOMATIS ---
# Fungsi ini berjalan sekali saat aplikasi dibuka untuk melatih model dari CSV Anda
@st.cache_resource
def train_model():
    csv_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # Cek apakah file ada
    if not os.path.exists(csv_file):
        return None, None, None, None, None

    # Load & Clean Data
    df = pd.read_csv(csv_file)
    df = df.drop('customerID', axis=1)
    
    # Konversi TotalCharges ke angka
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Mapping Target (Yes=1, No=0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # One-Hot Encoding (Mengubah kategori jadi angka 0/1)
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Simpan nama kolom fitur (penting untuk input user nanti)
    feature_names = df_encoded.drop('Churn', axis=1).columns.tolist()
    
    # Split Data
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- MODELING ---
    # 1. Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_scaled, y_train)
    
    # 2. Linear SVM (Calibrated agar keluar %)
    svm_base = LinearSVC(dual=False, random_state=42)
    svm = CalibratedClassifierCV(svm_base) 
    svm.fit(X_train_scaled, y_train)
    
    # 3. Ensemble (Gabungan)
    ensemble = VotingClassifier(estimators=[('nb', nb), ('svm', svm)], voting='soft')
    ensemble.fit(X_train_scaled, y_train)
    
    return ensemble, nb, svm, scaler, feature_names

# Load Model
model_ensemble, model_nb, model_svm, scaler, feature_names = train_model()

# --- FUNGSI UI BANTUAN ---
def get_confidence_badge(prob):
    if prob > 0.7:
        return "🔴 Tinggi (Bahaya)", "error"
    elif prob > 0.5:
        return "🟡 Sedang (Waspada)", "warning"
    else:
        return "🟢 Rendah (Aman)", "success"

# --- UI UTAMA ---
st.title("📉 Analisis Churn Pelanggan")
st.markdown("### Ensemble Model (Naive Bayes + SVM)")

if model_ensemble is None:
    st.error("⚠ File 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
else:
    st.subheader("✍ Data Profil Pelanggan")

    # Pilihan Preset Profil (Mirip "Pilih Contoh Ulasan")
    example_profiles = {
        "-- Input Manual --": None,
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

    selected_example = st.selectbox("Pilih contoh profil:", list(example_profiles.keys()))
    defaults = example_profiles[selected_example] if selected_example != "-- Input Manual --" else None

    # Form Input User
    with st.container():
        c1, c2 = st.columns(2)
        with c1:
            tenure = st.number_input("Lama Langganan (Bulan)", 0, 100, value=defaults['tenure'] if defaults else 12)
            contract = st.selectbox("Kontrak", ['Month-to-month', 'One year', 'Two year'], index=['Month-to-month', 'One year', 'Two year'].index(defaults['Contract']) if defaults else 0)
            monthly = st.number_input("Biaya Bulanan (USD)", 0.0, 200.0, value=defaults['MonthlyCharges'] if defaults else 50.0)
            tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(defaults['TechSupport']) if defaults else 0)

        with c2:
            internet = st.selectbox("Internet", ['DSL', 'Fiber optic', 'No'], index=['DSL', 'Fiber optic', 'No'].index(defaults['InternetService']) if defaults else 0)
            payment = st.selectbox("Pembayaran", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], index=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'].index(defaults['PaymentMethod']) if defaults else 0)
            total = st.number_input("Total Biaya (USD)", 0.0, 10000.0, value=defaults['TotalCharges'] if defaults else 100.0)
            security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'], index=['No', 'Yes', 'No internet service'].index(defaults['OnlineSecurity']) if defaults else 0)

    # Tombol Aksi
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary")
    with col2:
        show_comparison = st.checkbox("Bandingkan model", value=True)
    with col3:
        show_details = st.checkbox("Detail preprocessing", value=False)

    # Logic Prediksi
    if predict_btn:
        with st.spinner('Menganalisis pola data...'):
            try:
                # 1. Kumpulkan Input
                input_data = {
                    'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total,
                    'Contract': contract, 'InternetService': internet, 'PaymentMethod': payment,
                    'OnlineSecurity': security, 'TechSupport': tech_support,
                    # Nilai Default untuk fitur yang tidak ditampilkan di form agar simpel
                    'gender': 'Male', 'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes',
                    'MultipleLines': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
                    'StreamingTV': 'No', 'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'SeniorCitizen': 0
                }
                
                # 2. Preprocessing (Harus sama persis dengan training)
                input_df = pd.DataFrame([input_data])
                input_encoded = pd.get_dummies(input_df)
                # Reindex memaksa kolom input sama dengan kolom training (isi 0 jika fitur tidak ada)
                input_final = input_encoded.reindex(columns=feature_names, fill_value=0)
                input_scaled = scaler.transform(input_final)

                # 3. Prediksi
                pred_ens = model_ensemble.predict(input_scaled)[0]
                prob_ens = model_ensemble.predict_proba(input_scaled)[0] # [Prob No, Prob Yes]

                # === TAMPILAN HASIL ===
                st.subheader("🎯 Hasil Analisis (Ensemble)")
                
                churn_prob = prob_ens[1]
                conf_text, conf_type = get_confidence_badge(churn_prob)
                
                # Status Utama
                if pred_ens == 1:
                    st.error(f"### ❌ Status: AKAN CHURN (Berhenti)")
                else:
                    st.success(f"### ✅ Status: STAY (Setia)")
                
                st.info(f"Tingkat Risiko Churn: {conf_text} ({churn_prob*100:.1f}%)")

                # Metrics Probabilitas
                st.write("📊 Probabilitas:")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Akan Stay", f"{prob_ens[0]*100:.1f}%")
                with m2:
                    st.metric("Akan Churn", f"{prob_ens[1]*100:.1f}%")

                # Bandingkan Model
                if show_comparison:
                    st.divider()
                    st.subheader("📌 Perbandingan Model")
                    
                    # Hitung prediksi model lain
                    prob_nb = model_nb.predict_proba(input_scaled)[0][1]
                    prob_svm = model_svm.predict_proba(input_scaled)[0][1]
                    
                    k1, k2, k3 = st.columns(3)
                    with k1:
                        st.metric("Naive Bayes", f"{prob_nb*100:.1f}% Risk", "Prob. Churn")
                    with k2:
                        st.metric("Linear SVM", f"{prob_svm*100:.1f}% Risk", "Prob. Churn")
                    with k3:
                        st.metric("Ensemble", f"{prob_ens[1]*100:.1f}% Risk", "Prob. Churn")

                # Detail Preprocessing
                if show_details:
                    st.divider()
                    st.subheader("🔎 Detail Preprocessing")
                    st.text("Input Mentah:")
                    st.write(input_data)
                    st.text("Hasil One-Hot Encoding & Scaling (Input ke Model):")
                    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names).head())

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
