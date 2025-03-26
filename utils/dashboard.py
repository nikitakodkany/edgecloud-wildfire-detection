import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

st.set_page_config(layout="wide")
st.title("Wildfire Detection Dashboard")

uploaded_file = st.file_uploader("Upload a satellite or drone image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model():
    session = ort.InferenceSession("models/wildfire_model.onnx")
    return session, session.get_inputs()[0].name

def predict(img_np, session, input_name):
    input_tensor = np.transpose(img_np, (2, 0, 1))[np.newaxis].astype(np.float32) / 255.0
    pred = session.run(None, {input_name: input_tensor})[0]
    return (pred[0, 0] > 0.5).astype(np.uint8)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    img_np = np.array(image)

    st.image(image, caption="Input Image", use_column_width=True)

    session, input_name = load_model()
    mask = predict(img_np, session, input_name)

    overlay = img_np.copy()
    overlay[mask == 1] = [255, 0, 0]  # red overlay

    col1, col2 = st.columns(2)
    with col1:
        st.image(mask * 255, caption="Predicted Fire Mask", use_column_width=True)
    with col2:
        st.image(overlay, caption="Overlay", use_column_width=True)

    st.success("Detection complete!")