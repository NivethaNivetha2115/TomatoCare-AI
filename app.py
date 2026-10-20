import streamlit as st
import tensorflow as tf
import numpy as np
import requests
from PIL import Image
import os

st.set_page_config(page_title="TomatoCare AI", page_icon="🍅")

st.title("🍅 TomatoCare AI")
st.write("AI-powered Tomato Leaf Disease Detection")

# Google Drive model file ID
MODEL_FILE_ID = "1kWzuIVVZkfgdsFu0eGp0pa0DfBnpEY6y"
MODEL_PATH = "model.weights.h5"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}& confirm=t"
        response = requests.get(url, timeout=120)
        response.raise_for_status()

        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.load_weights(MODEL_PATH)
    return model


uploaded_file = st.file_uploader(
    "Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded Tomato Leaf",
        use_container_width=True
    )

    resized_image = image.resize((128, 128))
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    try:
        model = load_model()
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))

        st.subheader("Prediction Result")
        st.success(f"Predicted class number: {predicted_index}")
        st.write(
            f"Confidence: {predictions[predicted_index] * 100:.2f}%"
        )

    except Exception as error:
        st.error(f"Could not load the model: {error}")
