import os
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import json

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "ecg_efficientnetb0.h5"   # path to your .h5 model
IMG_SIZE = (224, 224)

# Load class indices mapping
with open("class_indices.json") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_ecg_model():
    model = load_model(MODEL_PATH)
    return model

model = load_ecg_model()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(uploaded_file, img_size=IMG_SIZE):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # force RGB
    img = cv2.resize(img, img_size)
    img = img.astype("float32")
    img = preprocess_input(img)  # ✅ EfficientNet preprocessing
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.set_page_config(page_title="ECG EfficientNet Classifier", page_icon="❤️", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-top: 15px;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


st.title("ECG Image Classification (EfficientNet)")

app_mode = st.sidebar.selectbox("Choose Mode", ["Home", "Upload & Classify", "About"])

# -----------------------------
# HOME PAGE
# -----------------------------
if app_mode == "Home":
    st.markdown("""
    ## Welcome 👋  
    This app classifies ECG images into:
    - Normal  
    - COVID-19  
    - Abnormal Heartbeat  
    - Myocardial Infarction (MI)  
    - MI History (PMI)  
    """)
    st.info("Go to **Upload & Classify** in the sidebar to test your own ECG images.")

# -----------------------------
# UPLOAD & CLASSIFY
# -----------------------------
elif app_mode == "Upload & Classify":
    uploaded_files = st.file_uploader(
        "Upload one or more ECG images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns(2)

            with col1:
                st.image(uploaded_file, caption="Uploaded ECG Image", use_container_width=True)

            with st.spinner("Processing..."):
                img = preprocess_image(uploaded_file)
                preds = model.predict(img)
                pred_idx = np.argmax(preds)
                pred_class = class_names[pred_idx]
                confidence = preds[0][pred_idx] * 100
                proba = (preds[0] * 100).round(2)

            with col2:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.subheader("Prediction Result")
                st.write(f"**Class:** {pred_class}")
                st.metric("Confidence", f"{confidence:.2f}%")

                # Top-3 predictions
                top3_idx = preds[0].argsort()[-3:][::-1]
                st.write("### Top 3 Predictions:")
                for i in top3_idx:
                    st.write(f"- {class_names[i]} ({preds[0][i]*100:.2f}%)")

                st.markdown('</div>', unsafe_allow_html=True)

            # Probability bar chart
            proba_df = {
                "Class": class_names,
                "Probability": proba
            }
            fig_bar = px.bar(
                proba_df,
                x="Probability",
                y="Class",
                orientation="h",
                title="Prediction Probabilities",
                text="Probability"
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

            # Probability pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=class_names,
                values=proba,
                hole=0.4,
                textinfo="label+percent",
                marker=dict(colors=px.colors.qualitative.Set3)
            )])
            fig_pie.update_layout(title="Prediction Probability Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# ABOUT PAGE
# -----------------------------
elif app_mode == "About":
    st.markdown("""
    ## About this App  
    - **Model:** EfficientNetB0 (fine-tuned)  
    - **Framework:** TensorFlow / Keras  
    - **Dataset:** ECG images (5 classes)  
    - **Goal:** Assist in automatic ECG interpretation  
    """)
