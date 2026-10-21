import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="TomatoCare AI",
    page_icon="🍅"
)

st.title("TomatoCare AI")
st.write("AI-powered Tomato Leaf Disease Detection")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.weights.h5"
)

# Disease names in the same order as the training dataset
class_names = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Healthy"
]


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.weights.h5 file not found in the repository."
        )

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            64, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(
            128, (3, 3), activation="relu"
        ),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
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

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    try:
        model = load_model()

        predictions = model.predict(
            image_array,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_disease = class_names[predicted_index]

        confidence = (
            predictions[predicted_index] * 100
        )

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Disease: {predicted_disease}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as error:
        st.error(
            f"Could not load the model: {error}"
        )
