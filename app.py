import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

treatments = {
    "Tomato Early Blight": {
        "cause": "Caused by fungus Alternaria solani",
        "symptoms": "Brown spots with yellow rings on lower leaves",
        "treatment": [
            "Remove and destroy all infected leaves immediately",
            "Spray with fungicide containing Chlorothalonil every 7 days",
            "Avoid watering leaves directly — water at the base only",
            "Apply copper-based fungicide as an alternative treatment"
        ],
        "prevention": [
            "Rotate crops every season — do not plant tomato in same spot",
            "Plant disease-resistant tomato varieties",
            "Ensure proper spacing between plants for air circulation",
            "Remove plant debris after harvest"
        ]
    },
    "Tomato Late Blight": {
        "cause": "Caused by water mold Phytophthora infestans",
        "symptoms": "Dark watery spots on leaves and stems, white mold underneath",
        "treatment": [
            "Remove and destroy infected plants immediately",
            "Spray with fungicide containing Mancozeb or Metalaxyl",
            "Apply treatment every 5-7 days during wet weather",
            "Do not compost infected plants — burn or bury them"
        ],
        "prevention": [
            "Avoid overhead irrigation — use drip irrigation",
            "Plant certified disease-free seeds",
            "Ensure good drainage in the farm",
            "Monitor crops regularly especially during rainy season"
        ]
    },
    "Tomato Healthy": {
        "cause": "No disease detected",
        "symptoms": "No symptoms — plant appears healthy",
        "treatment": [
            "No treatment needed",
            "Continue regular watering and fertilization",
            "Monitor regularly for early signs of disease"
        ],
        "prevention": [
            "Maintain good soil nutrition with balanced fertilizer",
            "Water consistently — avoid overwatering",
            "Keep farm clean and free of weeds",
            "Inspect plants regularly for early disease detection"
        ]
    }
}

# ── Website Interface ──────────────────────────────
st.title("🌿 Plant Disease Detection System")
st.subheader("Federal University of Technology Babura")
st.markdown("---")
st.write("📸 Upload a tomato leaf photo to detect disease and get treatment advice!")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", width=350)

    with st.spinner("Analysing image..."):
        img = image.resize((224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.markdown("---")

    # Result
    if result == "Tomato Healthy":
        st.success(f"✅ Result: {result}")
    else:
        st.error(f"⚠️ Disease Detected: {result}")

    st.info(f"📊 Confidence Score: {confidence:.2f}%")
    st.markdown("---")

    # Disease Info
    info = treatments[result]
    st.subheader("🔬 Disease Information")
    st.write(f"**Cause:** {info['cause']}")
    st.write(f"**Symptoms:** {info['symptoms']}")

    st.markdown("---")
    st.subheader("💊 Recommended Treatment")
    for t in info['treatment']:
        st.write(f"• {t}")

    st.markdown("---")
    st.subheader("🛡️ Prevention Tips")
    for p in info['prevention']:
        st.write(f"• {p}")

    st.markdown("---")
    st.caption("Developed by Yusuf Gambo | FUTB Computer Science | 2024/2025")
