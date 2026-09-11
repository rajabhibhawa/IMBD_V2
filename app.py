import numpy as np
import streamlit as st
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="IMDB Sentiment Classifier - v2",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment (v2)")
st.caption("Klasifikasi sentimen review film menggunakan model GRU")

MAX_LEN = 250  # sesuai MAX_LEN_B waktu training


# =========================================================
# PILIH MODEL  --> [FITUR BARU v2] bisa pilih antar model
# =========================================================
model_paths = {
    "GRU - Model 1": "best_model_imdb_GRU_Config2_SeqB.h5",
    "GRU - Model 2": "GRU_Config1_SeqB.h5"
}

model_choice = st.selectbox("Pilih Model", list(model_paths.keys()))


# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_sentiment_model(path):
    return load_model(path, compile=False)

try:
    model = load_sentiment_model(model_paths[model_choice])
except Exception as error:
    st.error(f"Gagal memuat model: {error}")
    st.stop()


# =========================================================
# LOAD TOKENIZER (SAMA PERSIS DENGAN YANG DIPAKAI TRAINING)
# =========================================================
@st.cache_resource
def load_tokenizer(path="tokenizer.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    tokenizer = load_tokenizer()
except Exception as error:
    st.error(f"Gagal memuat tokenizer: {error}")
    st.stop()


# =========================================================
# TEXT CLEANING (SAMA PERSIS DENGAN NOTEBOOK TRAINING)
# =========================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)          # hapus tag HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)          # hapus angka & simbol
    text = re.sub(r'\s+', ' ', text).strip()          # hapus spasi berlebih
    return text


def encode_review(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded


# =========================================================
# INPUT USER
# =========================================================
st.subheader("📝 Masukkan Review Film")

user_review = st.text_area(
    "Tulis review film (dalam Bahasa Inggris)",
    height=150,
    placeholder="Contoh: This movie was absolutely fantastic, great acting and story..."
)

if st.button("Analisis Sentimen", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Silakan masukkan review terlebih dahulu.")
    else:
        try:
            padded_review = encode_review(user_review)

            prediction = model.predict(padded_review, verbose=0)
            score = float(prediction[0][0])

            sentiment = "Positive 😊" if score > 0.5 else "Negative 😞"
            confidence = score if score > 0.5 else 1 - score

            # [FITUR BARU v2] tampilkan model yang dipakai
            st.write(f"### Sentimen: {sentiment}")
            st.write(f"Model digunakan: **{model_choice}**")

            # [FITUR BARU v2] confidence score + progress bar
            st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
            st.progress(float(confidence))

        except Exception as error:
            st.error(f"Prediksi gagal: {error}")