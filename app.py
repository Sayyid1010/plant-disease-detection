import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

st.title("🌿 Plant Disease Detection System")
st.write("Upload a tomato leaf photo to detect disease instantly!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf", width=300)
    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    result = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    st.success(f"Disease Detected: {result}")
    st.info(f"Confidence: {confidence:.2f}%")
