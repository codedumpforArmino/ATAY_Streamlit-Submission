import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

try:       
    model = load_model("streamlit_model.keras")
except Exception as e:
    st.error("Error loading model:" + e)

#vasc(6) = vascular Lesions, df(3) = dermatofibroma, bcc(1) = basal cell carcinoma, akiece (0) = actinic keratoses 
CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions', #(2)
    'Dermatofibroma',
    'Melanoma', #(4)
    'Melanocytic nevi', #(5)
    'Vascular lesions',
]

def preprocess_image(img_file):
    img = Image.open(img_file)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img, img_array

def predict(model, img_array):
    preds = model.predict(img_array)
    pred_class_index = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_class_index]
    predicted_class = CLASS_NAMES[pred_class_index]
    return predicted_class, confidence, preds[0]


st.markdown('<div class="main-title">🧬 Skin Disease Detection</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-info">Upload an image of a skin condition to get a prediction using a trained AI model.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('🔄 Processing image...'):
        img, img_array = preprocess_image(uploaded_file)
        predicted_class, confidence, class_probs = predict(model, img_array)

    st.image(img, caption='🖼 Uploaded Image', use_container_width=True)

    st.markdown(f'<div class="prediction">🔍 <strong>Prediction:</strong> {predicted_class}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="confidence">📊 Confidence: {confidence:.2f}</p>', unsafe_allow_html=True)

    with st.expander("🔎 See Raw Model Output"):
        st.write("Class probabilities array:", class_probs)

    st.markdown("### ℹ️ Class Information")
    st.info((predicted_class, "No additional details available."))