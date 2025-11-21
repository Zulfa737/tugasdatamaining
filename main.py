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

# --- 1. Konfigurasi Halaman ---
st.set_page_config(
    page_title="Analisis Churn Pelanggan",
    page_icon="📉",
    layout="centered"
)

# --- 2. Load Model & Tools (Otomatis Train dari CSV) ---
@st.cache_resource
def load_model_objects():
    csv_file = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(csv_file):
        return None, None, None, None, None

    try:
        # Load & Cleaning
        df = pd.read_csv(csv_file)
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        # Preprocessing
        df_encoded = pd.get_dummies(df, drop_first=True)
        feature_names = df_encoded.drop('Churn', axis=1).columns.tolist()
        
        # Split & Scale
        X = df_encoded.drop('Churn', axis=1)
        y = df_encoded['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Modeling
        # 1. Naive Bayes (Pengganti BernoulliNB di teks)
        model_nb = GaussianNB()
        model_nb.fit(X_train_scaled, y_train)
        
        # 2. SVM
        svm_base = LinearSVC(dual=False, random_state=42)
        model_svm = CalibratedClassifierCV(svm_base)
        model_svm.fit(X_train_scaled, y_train)
        
        # 3. Ensemble
        model_ensemble = VotingClassifier(estimators=[('nb', model_nb), ('svm', model_svm)], voting='soft')
        model_ensemble.fit(X_train_scaled, y_train)
        
        # Return semua object
        return model_nb, model_svm, model_ensemble, scaler, feature_names
        
    except Exception as e:
        st.error(f"Error Training: {e}")
        return None, None, None, None, None

# Load objects
model_nb, model_svm, model_ensemble, scaler, feature_names = load_model_objects()

# --- 3. Fungsi Bantuan (Helper) ---
# Preprocessing Input (Pengganti preprocess_text)
def preprocess_input(data, scaler, feature_names):
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)
    # Reindex agar kolom sama persis dengan data training (isi 0 jika tidak ada)
    df_final = df_encoded.reindex(columns=feature_names, fill_value=0)
    # Scaling
    df_scaled = scaler.transform(df_final)
    return df_scaled, df_final

# Confidence Label
def get_confidence_badge(prob):
    if prob > 75:
        return "🔴 Tinggi", "error"
    elif prob > 50:
        return "🟡 Sedang", "warning"
    else:
        return "🟢 Rendah", "success"

# --- 4. UI Utama ---
st.title("📉 Analisis Churn Pelanggan")
st.markdown("### Ensemble Model (Naive Bayes + SVM)")

models_loaded = all([model_nb, model_svm, model_ensemble, scaler, feature_names])

if not models_loaded:
    st.error("⚠ File CSV tidak ditemukan atau gagal dimuat. Pastikan 'WA_Fn-UseC_-Telco-Customer-Churn.csv' ada.")
else:
    st.subheader("✍ Masukkan Data Profil")

    # --- Pilihan Contoh (Seperti 'Pilih contoh ulasan') ---
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

    # --- Input Form (Pengganti Text Area) ---
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

    # --- Tombol Aksi (Sama Persis dengan Kode Film) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary")
    with col2:
        show_comparison = st.checkbox("Bandingkan model", value=True)
    with col3:
        show_details = st.checkbox("Detail preprocessing", value=False)

    if predict_btn:
        with st.spinner('Menganalisis...'):
            try:
                # 1. Siapkan Data Input Dictionary
                input_data = {
                    'tenure': tenure, 'MonthlyCharges': monthly, 'TotalCharges': total,
                    'Contract': contract, 'InternetService': internet, 'PaymentMethod': payment,
                    'OnlineSecurity': security, 'TechSupport': tech,
                    # Default values untuk fitur yang tidak ditampilkan agar UI bersih
                    'gender': 'Male', 'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes',
                    'MultipleLines': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No',
                    'StreamingTV': 'No', 'StreamingMovies': 'No', 'PaperlessBilling': 'Yes', 'SeniorCitizen': 0
                }

                # 2. Preprocess
                vec, df_processed = preprocess_input(input_data, scaler, feature_names)

                # 3. Prediksi
                pred_nb = model_nb.predict(vec)[0]
                pred_svm = model_svm.predict(vec)[0]
                pred_ensemble = model_ensemble.predict(vec)[0]

                prob_nb = model_nb.predict_proba(vec)[0]
                prob_svm = model_svm.predict_proba(vec)[0]
                prob_ensemble = model_ensemble.predict_proba(vec)[0]

                # ======== Hasil Ensemble ========
                st.subheader("🎯 Hasil Analisis (Ensemble)")

                # Class 1 = Churn, Class 0 = Stay
                churn_prob = prob_ensemble[1] * 100
                conf_text, conf_type = get_confidence_badge(churn_prob)

                if pred_ensemble == 1:
                    st.error("### ❌ Prediksi: AKAN CHURN (Berhenti)")
                    st.write("Pelanggan ini berisiko tinggi untuk berhenti berlangganan.")
                else:
                    st.success("### ✅ Prediksi: SETIA (Stay)")
                    st.write("Pelanggan ini diprediksi akan tetap berlangganan.")

                st.info(f"Tingkat Risiko Churn: {conf_text} ({churn_prob:.1f}%)")

                # Probabilitas
                st.write("📊 Probabilitas:")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Akan Stay", f"{prob_ensemble[0]*100:.1f}%")
                with col2:
                    st.metric("Akan Churn", f"{prob_ensemble[1]*100:.1f}%")

                # ======== Bandingkan Model ========
                if show_comparison:
                    st.subheader("📌 Perbandingan Model")

                    comp1, comp2, comp3 = st.columns(3)
                    with comp1:
                        st.metric("Naive Bayes", 
                                  "CHURN" if pred_nb == 1 else "STAY",
                                  f"{prob_nb[1]*100:.1f}% Risk")
                    with comp2:
                        st.metric("Linear SVM", 
                                  "CHURN" if pred_svm == 1 else "STAY",
                                  f"{prob_svm[1]*100:.1f}% Risk")
                    with comp3:
                        st.metric("Ensemble", 
                                  "CHURN" if pred_ensemble == 1 else "STAY",
                                  f"{prob_ensemble[1]*100:.1f}% Risk")

                # ======== Detail Preprocessing ========
                if show_details:
                    st.subheader("🔎 Detail Preprocessing")
                    st.text("Input Mentah (Dictionary):")
                    st.write(input_data)
                    st.text("Setelah Encoding & Scaling (Siap Masuk Model):")
                    st.dataframe(pd.DataFrame(vec, columns=feature_names).head())

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
