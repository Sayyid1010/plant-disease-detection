import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("plant_disease_model.h5")
classes = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

st.title("🌿 Plant Disease Detection System")
st.write("Upload a tomato leaf photo to detect the disease instantly!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", width=300)
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    st.success(f"Disease Detected: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
