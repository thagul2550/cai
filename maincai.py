import os
import io
import streamlit as st
import gdown
from keras.models import load_model
from PIL import Image
import pandas as pd
from util import classify, set_background
from datetime import datetime
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import ConvNeXtTiny

MODEL_PATH = "model/convnext_best.h5"
LABELS_PATH = "model/labels.txt"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1AiTLRufmeh-DKJlwvWvBXrF53X_0z6ZF"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading model from Google Drive...")
        os.makedirs("model", exist_ok=True)
        gdown.download(GOOGLE_DRIVE_URL, output=MODEL_PATH, quiet=False)
        st.success("✅ Model downloaded successfully!")

class DepthwiseConv2DFixed(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

_ = ConvNeXtTiny()  # ลงทะเบียน ConvNeXt
get_custom_objects().update({
    'DepthwiseConv2D': DepthwiseConv2DFixed,
    'ConvNeXt': ConvNeXtTiny
})

@st.cache_resource
def load_model_cached():
    download_model()
    custom_objects = {'DepthwiseConv2D': DepthwiseConv2DFixed, 'ConvNeXt': ConvNeXtTiny}
    model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    return model

def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error("❌ Labels file not found! Please upload labels.txt to model/")
        return []
    with open(LABELS_PATH, 'r') as f:
        return [line.strip().split()[-1] for line in f]

# UI Setup
st.set_page_config(page_title="CP ALL", page_icon="📊", layout="centered", initial_sidebar_state="expanded")

try:
    set_background("bg/AI for preventive maintenance Signage of 7-ELEVEN's with photo inspection.png")
except Exception as e:
    st.error(f"Error setting background: {e}")

st.markdown(
    '<div style="text-align: center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/7-eleven_logo.svg/791px-7-eleven_logo.svg.png" width="150"></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align: center; color: white;"><h1>AI for Preventive Maintenance Signage of 7-ELEVEN with Photo Inspection</h1></div>',
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    .stTextInput label, .stSlider label, .stFileUploader label { color: white; }
    </style>
""", unsafe_allow_html=True)

name = st.text_input("Employee name:")
code = st.text_input("Branch code:")
sign_type = st.selectbox("Select Sign Type:", ["Pole Sign", "Fin Sign"])
many = st.slider("How many pictures:", 1, 6)
model_type = st.radio("Select Part", ("Signs", "Base"))

try:
    model = load_model_cached()
    class_names = load_labels()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

excel_file_path = 'data.xlsx'
image_folder_path = 'images/'
if not os.path.exists(image_folder_path):
    os.makedirs(image_folder_path)

files = st.file_uploader("Upload images", type=['jpeg', 'jpg', 'png', 'jfif'], accept_multiple_files=True)

if files:
    image_data = []
    upload_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for i, file in enumerate(files):
        image = Image.open(file).convert('RGB')
        st.image(image, use_column_width=True)

        try:
            class_name, confidence_score = classify(image, model, class_names)
            image_name = f"{code}_{class_name}_{i+1}.png"
            image_path = os.path.join(image_folder_path, image_name)
            image.save(image_path)

            st.write(f"## {class_name} (Confidence: {confidence_score:.2f})")
            st.write(f"### Your sign is classified as {class_name}")

            note = st.text_area(f"Note {i+1}", key=f"note_{i+1}")

            image_data.append({
                'Image': image_name,
                'Phase': class_name,
                'Confidence': f"{confidence_score:.2f}",
                'Upload Time': upload_time,
                'Note': note
            })
        except Exception as e:
            st.error(f"Error during classification: {e}")

    if st.button('Submit'):
        df = pd.DataFrame(image_data)
        df['Employee name'] = name
        df['Branch code'] = code
        df['Sign type'] = sign_type
        df['How many images'] = many
        df['Part'] = model_type

        try:
            if os.path.exists(excel_file_path):
                existing_df = pd.read_excel(excel_file_path)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                updated_df = df

            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='w') as writer:
                updated_df.to_excel(writer, index=False)

            st.success('✅ Submission complete')

            # เตรียมไฟล์ Excel สำหรับดาวน์โหลดแบบ BytesIO
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                updated_df.to_excel(writer, index=False)
            towrite.seek(0)

            st.download_button(
                label="Download data as Excel",
                data=towrite,
                file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Error saving Excel file: {e}")
