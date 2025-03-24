import streamlit as st
import zipfile
import os
import shutil
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("violence_detector_model.keras")
class_labels = ['Normal', 'Violence']

st.title("🔍 Violence Detection from CCTV Frames")
st.write("Upload a zip file containing 5 JPG frames (0.jpg to 4.jpg)")

# File uploader
uploaded_file = st.file_uploader("Upload ZIP of frames", type="zip")

if uploaded_file:
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall("temp_frames")

    results = []

    for i in range(5):
        frame_path = os.path.join("temp_frames", f"{i}.jpg")
        if not os.path.exists(frame_path):
            st.warning(f"Frame {i}.jpg is missing!")
            continue

        # Load and preprocess
        img = cv2.imread(frame_path)
        img = cv2.resize(img, (128, 128))
        img = img.astype("float32") / 255.0
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)
        pred_class = np.argmax(pred)
        confidence = pred[0][pred_class]

        results.append((f"{i}.jpg", class_labels[pred_class], confidence))

    # Display results
    for frame, label, score in results:
        st.success(f"🖼️ {frame} → **{label}** (Confidence: {score:.2f})")

    # Cleanup
    shutil.rmtree("temp_frames")
