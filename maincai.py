import streamlit as st
from keras.models import load_model
from PIL import Image
import pandas as pd
import os
from util import classify, set_background
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.applications import ConvNeXtTiny

# ✅ แก้ DepthwiseConv2DFixed
class DepthwiseConv2DFixed(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# ✅ ลงทะเบียน custom layers
_ = ConvNeXtTiny()  # ลงทะเบียน ConvNeXt
get_custom_objects().update({
    'DepthwiseConv2D': DepthwiseConv2DFixed,
    'ConvNeXt': ConvNeXtTiny
})

# ---------------------------- Streamlit UI ----------------------------

st.set_page_config(
    page_title="CP ALL",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ✅ Set background
try:
    set_background("bg/AI for preventive maintenance Signage of 7-ELEVEN's with photo inspection.png")
except Exception as e:
    st.error(f"Error setting background: {e}")

# ✅ โลโก้และหัวข้อ
st.markdown(
    '<div style="text-align: center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/40/7-eleven_logo.svg/791px-7-eleven_logo.svg.png" width="150"></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align: center; color: white;"><h1>AI for Preventive Maintenance Signage of 7-ELEVEN with Photo Inspection</h1></div>',
    unsafe_allow_html=True
)

# ✅ สไตล์ input
st.markdown("""
    <style>
    .stTextInput label, .stSlider label, .stFileUploader label { color: white; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- Input Form ----------------------------

name = st.text_input("Employee name:")
code = st.text_input("Branch code:")
sign_type = st.selectbox("Select Sign Type:", ["Pole Sign", "Fin Sign"])
many = st.slider("How many pictures:", 1, 6)
model_type = st.radio("Select Part", ("Signs", "Base"))

model_paths = {
    "Base": "model/convnext_best.h5"  # ✅ ใช้ ConvNeXt model
}
labels_paths = {
    "Base": "model/labels.txt"
}

model_path = model_paths[model_type]
labels_path = labels_paths[model_type]

# ✅ โหลดโมเดลด้วย custom_objects
try:
    custom_objects = {
        'DepthwiseConv2D': DepthwiseConv2DFixed,
        'ConvNeXt': ConvNeXtTiny
    }
    model = load_model(model_path, custom_objects=custom_objects, compile=False)

    with open(labels_path, 'r') as f:
        class_names = [line.strip().split()[-1] for line in f]

except Exception as e:
    st.error(f"Error loading model or labels: {e}")

# ---------------------------- Upload Images ----------------------------

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

            st.success('Submission complete')

            st.download_button(
                label="Download data as Excel",
                data=updated_df.to_excel(index=False),
                file_name=f"maintenance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"An error occurred while saving the file: {e}")
