import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="TomatoCare AI", page_icon="🍅")

st.title("🍅 TomatoCare AI")
st.write("AI-powered Tomato Leaf Disease Detection")

@st.cache_resource
def load_model():
return tf.keras.models.load_model("tomatocare_model.keras")

uploaded_file = st.file_uploader(
"Upload a tomato leaf image",
type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Tomato Leaf", use_container_width=True)

resized_image = image.resize((128, 128))
image_array = np.array(resized_image, dtype=np.float32)
image_array = np.expand_dims(image_array, axis=0)

try:
    model = load_model()
    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))

    st.subheader("Prediction Result")
    st.success(f"Predicted class number: {predicted_index}")
    st.write(f"Confidence: {predictions[predicted_index] * 100:.2f}%")

    st.info(
        "This is an educational prototype. "
        "The prediction may not always be correct."
    )

except Exception as error:
    st.error(f"Could not load the model: {error}")
