import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL
# ===============================
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans.pkl")
logreg = joblib.load("logreg.pkl")
features = joblib.load("features.pkl")

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(
    page_title="Air Quality Clustering",
    page_icon="🌫️",
    layout="centered"
)

st.title("🌫️ Air Quality Clustering & Classification")
st.markdown("""
Aplikasi ini menggunakan:
- **K-Means** untuk clustering kualitas udara
- **Logistic Regression** untuk klasifikasi cluster
""")

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("📊 Input Parameter Kualitas Udara")

user_input = {}

for feature in features:
    user_input[feature] = st.sidebar.number_input(
        feature,
        value=0.0,
        format="%.3f"
    )

input_df = pd.DataFrame([user_input])

# ===============================
# PREPROCESSING
# ===============================
input_scaled = scaler.transform(input_df)

# ===============================
# PREDIKSI
# ===============================
if st.button("🔍 Prediksi Cluster"):
    cluster_kmeans = kmeans.predict(input_scaled)[0]
    cluster_logreg = logreg.predict(input_scaled)[0]
    prob = logreg.predict_proba(input_scaled)[0]

    st.subheader("📌 Hasil Prediksi")

    st.success(f"🔹 Cluster (K-Means): **Cluster {cluster_kmeans}**")
    st.info(f"🔹 Cluster (Logistic Regression): **Cluster {cluster_logreg}**")

    st.subheader("📈 Probabilitas Cluster (Logistic Regression)")
    prob_df = pd.DataFrame(
        prob.reshape(1, -1),
        columns=[f"Cluster {i}" for i in logreg.classes_]
    )
    st.dataframe(prob_df)

    # ===============================
    # INTERPRETASI CLUSTER
    # ===============================
    st.subheader("🧠 Interpretasi Cluster")

    if cluster_logreg == 0:
        st.write("""
        **Cluster 0 – Kualitas Udara Sedang**
        - Konsentrasi gas relatif stabil
        - Suhu dan kelembapan cukup tinggi
        """)
    elif cluster_logreg == 1:
        st.write("""
        **Cluster 1 – Kualitas Udara Buruk**
        - Konsentrasi polutan tinggi
        - Berpotensi berdampak buruk bagi kesehatan
        """)
    else:
        st.write("""
        **Cluster 2 – Kualitas Udara Baik**
        - Polutan rendah
        - Kondisi udara relatif bersih
        """)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("📌 Model: K-Means + Logistic Regression | Dataset Air Quality")

