import numpy as np
from scipy.special import softmax
import streamlit as st
import pickle
import pandas as pd
import io
import matplotlib.pyplot as plt


# Page config & Dark mode styling

st.set_page_config(page_title="Multinomial Logistic Text Classifier", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    div[data-testid="stMarkdownContainer"] {
        color: white;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div > div {
        background-color: #1c1f26;
        color: white;
    }
    button[kind="primary"] {
        background-color: #2563eb;
        color: white;
    }
    .stDataFrame thead tr th {
        background-color: #1c1f26;
        color: white;
    }
    .stDataFrame tbody tr td {
        color: white;
        background-color: #1c1f26;
    }
    canvas {
        background-color: #0e1117 !important;
    }
    .element-container .stAlert {
        background-color: #20232a;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)


# Model Class

class MultinomialLogisticRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def fit(self, X_sparse, y, iterations=1000):
        self.X = X_sparse
        self.classes = np.unique(y)
        n_samples, n_features = X_sparse.shape
        n_classes = len(self.classes)

        y_encoded = np.zeros((n_samples, n_classes))
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes)}
        for i, label in enumerate(y):
            y_encoded[i, class_to_index[label]] = 1

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        for _ in range(iterations):
            scores = X_sparse @ self.W + self.b
            probs = softmax(scores, axis=1)
            error = probs - y_encoded
            dW = X_sparse.T @ error / n_samples
            db = np.sum(error, axis=0, keepdims=True) / n_samples
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

    def predict(self, X_sparse):
        scores = X_sparse @ self.W + self.b
        probs = softmax(scores, axis=1)
        class_indices = np.argmax(probs, axis=1)
        return [self.classes[i] for i in class_indices]


# Load model dan vectorizer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Streamlit UI

st.title("🧠 Text Classification App")
st.caption("Pengantar Pemrosesan Data dan Multimedia")
st.markdown("---")

st.subheader("📄 Masukkan Teks yang Ingin Diklasifikasikan")
user_input = st.text_area("Teks Input", height=150, placeholder="Contoh: Produk ini sangat bagus dan berkualitas...")

true_label = st.selectbox("✅ Pilih Label Sebenarnya (Opsional)", options=[""] + list(model.classes))

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Prediksi"):
    if user_input.strip() == "":
        st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
    else:
        with st.spinner('Model sedang memproses...'):
            input_vec = vectorizer.transform([user_input])
            scores = input_vec @ model.W + model.b
            probs = softmax(scores, axis=1).flatten()
            predicted_class = model.classes[np.argmax(probs)]

        st.session_state.history.append((user_input, predicted_class))

        if true_label and true_label != predicted_class:
            st.error(f"❌ Prediksi salah. Prediksi Kamu: **{true_label}**, Seharusnya yang benar: **{predicted_class}**")
        else:
            st.success(f"🎯 Hasil Prediksi: **{predicted_class}**")

            prob_df = pd.DataFrame({
                "Kelas": model.classes,
                "Probabilitas": probs
            }).sort_values("Probabilitas", ascending=True)

            st.markdown("---")
            st.subheader("📊 Visualisasi Probabilitas") #menmapilkan gambar visualisasi

            fig, ax = plt.subplots()
            ax.barh(prob_df["Kelas"], prob_df["Probabilitas"])
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.set_xlabel("Probabilitas", color='white')
            ax.set_ylabel("Kelas", color='white')
            ax.set_title("Distribusi Probabilitas", color='white')
            st.pyplot(fig)

            csv = prob_df.reset_index().to_csv(index=False)
            st.download_button("📥 Unduh Hasil (CSV)", data=csv, file_name="hasil_prediksi.csv", mime="text/csv")


# Riwayat Prediksi

if len(st.session_state.history) > 0: #menampilkan riwayat prediksi
    st.markdown("---")
    st.subheader("🕓 Riwayat Prediksi")
    for i, (text, pred) in enumerate(st.session_state.history[::-1][:5]):
        st.markdown(f"**{i+1}.** _{text[:40]}..._ → **{pred}**")


# Footer
st.markdown("---")
st.caption("Made with ❤️ by Kelompok 2 - Kelas C")











