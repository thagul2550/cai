import os
import io
import streamlit as st
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import ConvNeXtTiny
from util import classify, set_background
from datetime import datetime

# ==========================
# 🔥 CONFIG
# ==========================
REPO_ID = "Gusgoodmansamadayo/Convnex_Base-7_11_Sign"
MODEL_FILE = "convnext_best.h5"
LABEL_FILE = "labels.txt"

# ==========================
# 🔥 Load Model & Labels from Hugging Face Hub
# ==========================
@st.cache_resource
def load_model_from_hf():
    token = os.getenv("HF_TOKEN", None)  # ใช้ token ถ้า repo เป็น private

    # ✅ ดาวน์โหลดไฟล์จาก Hugging Face Hub
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, token=token)
    labels_path = hf_hub_download(repo_id=REPO_ID, filename=LABEL_FILE, token=token)

    # ✅ Fix ConvNeXt custom layer
    class DepthwiseConv2DFixed(OriginalDepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            if 'groups' in kwargs:
                kwargs.pop('groups')
            super().__init__(*args, **kwargs)

    get_custom_objects().update({
        'DepthwiseConv2D': DepthwiseConv2DFixed,
        'ConvNeXt': ConvNeXtTiny
    })

    # ✅ โหลดโมเดล
    model = load_model(model_path, custom_objects={'DepthwiseConv2D': DepthwiseConv2DFixed, 'ConvNeXt': ConvNeXtTiny})

    # ✅ โหลด labels
    with open(labels_path, 'r') as f:
        class_names = [line.strip().split()[-1] for line in f]

    return model, class_names

# ==========================
# 🔥 UI Setup
# ==========================
st.set_page_config(page_title="CP ALL", page_icon="📊", layout="centered", initial_sidebar_state="expanded")

try:
    set_background("bg/AI for preventive maintenance Signage of 7-ELEVEN's with photo inspection.png")
except:
    st.warning("Background not found (optional).")

st.markdown(
    '<div style="text-align:center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/7-eleven_logo.svg/791px-7-eleven_logo.svg.png" width="150"></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center; color:white;"><h1>AI for Preventive Maintenance Signage of 7-ELEVEN with Photo Inspection</h1></div>',
    unsafe_allow_html=True
)
st.markdown("<style>.stTextInput label, .stSlider label, .stFileUploader label { color:white; }</style>", unsafe_allow_html=True)

# ==========================
# 🔥 User Inputs
# ==========================
name = st.text_input("Employee name:")
code = st.text_input("Branch code:")
sign_type = st.selectbox("Select Sign Type:", ["Pole Sign", "Fin Sign"])
many = st.slider("How many pictures:", 1, 6)
model_type = st.radio("Select Part", ("Signs", "Base"))

# ==========================
# 🔥 Load Model
# ==========================
try:
    model, class_names = load_model_from_hf()
    st.success("✅ Model loaded successfully from Hugging Face!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ==========================
# 🔥 File Upload + Classification
# ==========================
files = st.file_uploader("Upload images", type=['jpeg', 'jpg', 'png', 'jfif'], accept_multiple_files=True)
image_data = []

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

            image_data.append({
                'Image': f"uploaded_{i+1}.png",
                'Phase': class_name,
                'Confidence': f"{confidence_score:.2f}",
                'Upload Time': upload_time,
                'Note': note
            })
        except Exception as e:
            st.error(f"Error during classification: {e}")

    # ==========================
    # 🔥 Save to Excel in Memory
    # ==========================
    if st.button("Submit"):
        df = pd.DataFrame(image_data)
        df['Employee name'] = name
        df['Branch code'] = code
        df['Sign type'] = sign_type
        df['How many images'] = many
        df['Part'] = model_type

        towrite = io.BytesIO()
        with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        towrite.seek(0)

        st.success("✅ Submission complete")
        st.download_button(
            label="Download data as Excel",
            data=towrite,
            file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
