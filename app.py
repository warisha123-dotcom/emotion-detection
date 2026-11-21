import pickle
import numpy as np
import tensorflow as tf
import streamlit as st
import time

st.set_page_config(page_title="Emotion Detector 💖", page_icon="💖", layout="centered")

# --- CSS for Animation ---
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #FFE3EC, #E0C3FC, #B5EAD7); }
.big-emoji { font-size: 90px; text-align: center; animation: bounce 1s infinite; }
@keyframes bounce { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-15px); } }
.main-box { background: white; padding: 25px; border-radius: 20px; box-shadow: 0px 8px 20px rgba(0,0,0,0.15); text-align: center; }
.stButton>button { background: linear-gradient(90deg, #FF4B8B, #FF8A5C); color: white; font-size: 20px; border-radius: 15px; padding: 10px 25px; border: none; }
</style>
""", unsafe_allow_html=True)

# --- Load Files ---
@st.cache_resource
def load_all():
    with open("vectorizer_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("emotions.pkl", "rb") as f:
        emotions = pickle.load(f)
    model = tf.keras.models.load_model("emotion_bilstm.keras")
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=50, vocabulary=vocab)
    return vectorizer, emotions, model

vectorizer, emotions, model = load_all()

emotion_info = {
    "joy": ("😊", "#FFD93D", "Yay! You are feeling so Happy today! ✨"),
    "love": ("😍", "#FF6B81", "Aww! Your heart is full of Love! ❤️"),
    "anger": ("😡", "#FF6B6B", "Take a deep breath... You sound Angry! 😤"),
    "fear": ("😨", "#74C0FC", "Don't worry, I'm here! You sound Scared! 🫂"),
    "sadness": ("😢", "#A5B4FC", "It's okay to feel sad... Sending you a hug! 🤗"),
    "surprise": ("😲", "#C084FC", "WOW! That was really Surprising! 🤯"),
    "neutral": ("😐", "#94A3B8", "Hmm... You sound calm and Neutral! 😌")
}

# --- Title ---
st.markdown("<h1 style='text-align:center; color:#FF2E7E;'>💖 Emotion Detector 💖</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>✨ Type a sentence and let AI feel your emotion! ✨</p>", unsafe_allow_html=True)

st.markdown("<div class='main-box'><h3>🎤 How are you feeling today?</h3></div>", unsafe_allow_html=True)
st.write("")

# --- Example Buttons ---
st.write("👇 **Try One Click Examples:**")
c1, c2, c3, c4 = st.columns(4)
if c1.button("😊 Happy"):
    st.session_state.text = "I am so happy today, this is the best day!"
if c2.button("❤️ Love"):
    st.session_state.text = "I love my family so much"
if c3.button("😡 Angry"):
    st.session_state.text = "I am really angry right now"
if c4.button("😢 Sad"):
    st.session_state.text = "I feel so sad and alone today"

text = st.text_input("💬 Type your sentence here...", key="text", placeholder="e.g. I am very excited for my presentation...")

if st.button("✨
