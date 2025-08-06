import os
import streamlit as st
from PIL import Image
from huggingface_hub import snapshot_download
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import ConvNeXtTiny
from util import classify, set_background
from datetime import datetime

REPO_ID = "Gusgoodmansamadayo/Convnex_Base-7_11_Sign"
MODEL_FILE = "convnext_best.h5"
LABEL_FILE = "labels.txt"

@st.cache_resource
def load_model_from_hf():
    token = os.getenv("HF_TOKEN", None)

    model_dir = snapshot_download(repo_id=REPO_ID, token=token)
    model_path = os.path.join(model_dir, MODEL_FILE)
    labels_path = os.path.join(model_dir, LABEL_FILE)

    class DepthwiseConv2DFixed(OriginalDepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            if 'groups' in kwargs:
                kwargs.pop('groups')
            super().__init__(*args, **kwargs)

    get_custom_objects().update({
        'DepthwiseConv2D': DepthwiseConv2DFixed,
        'ConvNeXt': ConvNeXtTiny
    })

    model = load_model(model_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2DFixed, 'ConvNeXt': ConvNeXtTiny})

    with open(labels_path, 'r') as f:
        class_names = [line.strip().split()[-1] for line in f]

    return model, class_names

st.set_page_config(page_title="CP ALL", page_icon="📊", layout="centered")

try:
    set_background("bg/AI for preventive maintenance Signage of 7-ELEVEN's with photo inspection.png")
except Exception:
    pass

st.markdown(
    '<div style="text-align:center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/7-eleven_logo.svg/791px-7-eleven_logo.svg.png" width="150"></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center; color:white;"><h1>AI for Preventive Maintenance Signage of 7-ELEVEN with Photo Inspection</h1></div>',
    unsafe_allow_html=True
)
st.markdown("<style>.stTextInput label, .stSlider label, .stFileUploader label { color:white; }</style>", unsafe_allow_html=True)

name = st.text_input("Employee name:")
code = st.text_input("Branch code:")
sign_type = st.selectbox("Select Sign Type:", ["Pole Sign", "Fin Sign"])
many = st.slider("How many pictures:", 1, 6)
model_type = st.radio("Select Part", ("Signs", "Base"))

try:
    model, class_names = load_model_from_hf()
    st.success("✅ Model loaded successfully from Hugging Face!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

files = st.file_uploader("Upload images", type=['jpeg', 'jpg', 'png', 'jfif'], accept_multiple_files=True)

if files:
    upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i, file in enumerate(files):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        try:
            class_name, confidence_score = classify(image, model, class_names)
            st.write(f"## {class_name} (Confidence: {confidence_score:.2f})")
            st.write(f"### Your sign is classified as {class_name}")
            note = st.text_area(f"Note {i+1}", key=f"note_{i+1}")
        except Exception as e:
            st.error(f"Error during classification: {e}")
