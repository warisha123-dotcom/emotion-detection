import pickle
import numpy as np
import tensorflow as tf
import streamlit as st

with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

with open("emotions.pkl", "rb") as f:
    emotions = pickle.load(f)

model = tf.keras.models.load_model("emotion_bilstm.keras")

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=50,
    vocabulary=vocab
)

st.title("Emotion Detection from Text")
st.write("BiLSTM model trained on dair-ai/emotion dataset")

text = st.text_input("Type an English sentence here...")

if st.button("Predict Emotion"):
    if text:
        x = vectorizer([text]).numpy()
        probs = model.predict(x, verbose=0)[0]
        result = {emotions[i]: float(probs[i]) for i in range(len(emotions))}
        top = max(result, key=result.get)
        st.success(f"Predicted Emotion: {top}")
        st.bar_chart(result)
    else:
        st.warning("Please type a sentence first!")
