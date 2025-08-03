import os
import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    Set the background of a Streamlit app safely from the given image path.
    Supports Windows paths with backslashes, spaces, and special characters.
    """
    try:
        # ✅ ป้องกันปัญหา escape character โดย normalize path
        safe_path = os.path.normpath(image_file)

        if not os.path.exists(safe_path):
            st.warning(f"Background image not found: {safe_path}")
            return

        with open(safe_path, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()

        style = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-attachment: fixed;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error setting background: {e}")


def classify(image, model, class_names):
    """
    Classify an image using a trained Keras model and return class name + confidence score.
    """
    # Resize image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert to numpy
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Prepare data
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
