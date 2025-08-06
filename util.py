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
    Classify an image using a ConvNeXt or standard Keras model.
    Automatically detects model input size.
    """
    try:
        # ✅ Detect model input shape (ConvNeXt uses 224 or 256)
        input_shape = model.input_shape[1:3] if model.input_shape else (224, 224)
        target_size = (input_shape[0], input_shape[1])

        # ✅ Resize image
        image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

        # ✅ Normalize to [0,1]
        img_array = np.asarray(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # ✅ Predict
        prediction = model.predict(img_array)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = float(prediction[0][index])

        return class_name, confidence_score

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Unknown", 0.0
