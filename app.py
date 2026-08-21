import os
# Set Keras backend to PyTorch
os.environ["KERAS_BACKEND"] = "torch"

import streamlit as st
import gdown
import keras
import numpy as np
from PIL import Image

# Google Drive file ID 
FILE_ID = '117Htqb7ZTpB8FOK7-cMJEB_rlBV5zNFP'
MODEL_PATH = 'berkeley_vgg16_transfer_learning_model.keras'

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            
        with st.spinner("Downloading model from Google Drive, please wait..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
            
    return keras.models.load_model(MODEL_PATH)

# Load the model
with st.spinner("Preparing the model, please wait..."):
    model = load_my_model()

# User Interface
st.set_page_config(page_title="Berkeley Image Segmentation App", page_icon="🖼️", layout="centered")

st.title("Berkeley Segmentation with VGG16 Transfer Learning")
st.write("Please upload an image to perform pixel-level object segmentation.")

# File uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    if st.button("Perform Segmentation"):
        with st.spinner("Model is analyzing the image..."):
            try:
                # Resize image to match model's expected input shape (176x176)
                IMG_SIZE = 176
                img_resized = image.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img_resized, dtype=np.float32) / 255.0  # Normalization
                
                # Add batch dimension to match expected shape: (1, 176, 176, 3)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Perform prediction
                predictions = model.predict(img_array)
                pred_mask = predictions[0, :, :, 0]  # (176, 176)
                
                # Binary Thresholding
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                
                # Resize mask back to original image dimensions
                pred_mask_pil = Image.fromarray(pred_mask).resize(image.size, Image.NEAREST)
                
                st.success("Segmentation Complete!")
                
                with col2:
                    st.subheader("Segmentation Mask")
                    st.image(pred_mask_pil, use_container_width=True, clamp=True)
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
