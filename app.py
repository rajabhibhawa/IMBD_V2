import numpy as np
import streamlit as st

from tensorflow.keras.datasets import imdb
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


# =========================================================
# PILIH MODEL  --> [FITUR BARU v2] bisa pilih antar model
# =========================================================
model_paths = {
    "GRU - Model 1": "best_model_imdb_GRU_Config2_seqB.h5",
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
# LOAD WORD INDEX (BAWAAN DATASET IMDB KERAS)
# =========================================================
@st.cache_resource
def get_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index

word_index = get_word_index()

MAX_LEN = model.input_shape[1]


# =========================================================
# FUNGSI KONVERSI TEKS -> SEQUENCE ANGKA
# =========================================================
def encode_review(text):
    words = text.lower().split()
    encoded = [1]
    for word in words:
        idx = word_index.get(word, 2)
        if idx < 10000:
            encoded.append(idx)
        else:
            encoded.append(2)
    return encoded


# =========================================================
# INPUT USER
# =========================================================
st.subheader("📝 Masukkan Review Film")

user_review = st.text_area(
    "Tulis review film (dalam Bahasa Inggris, sesuai dataset IMDB)",
    height=150,
    placeholder="Contoh: This movie was absolutely fantastic, great acting and story..."
)

if st.button("Analisis Sentimen", type="primary", use_container_width=True):
    if not user_review.strip():
        st.warning("Silakan masukkan review terlebih dahulu.")
    else:
        try:
            encoded_review = encode_review(user_review)
            padded_review = pad_sequences([encoded_review], maxlen=MAX_LEN)

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