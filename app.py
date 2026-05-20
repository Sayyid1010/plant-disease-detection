import streamlit as st
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# ── Load Model ─────────────────────────────────────
interpreter = Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Tomato Early Blight", "Tomato Late Blight", "Tomato Healthy"]

treatments = {
    "Tomato Early Blight": {
        "emoji": "🟠",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Brown spots with yellow rings on lower leaves",
        "treatment": [
            "Remove and destroy all infected leaves immediately",
            "Spray with fungicide containing Chlorothalonil every 7 days",
            "Avoid watering leaves directly — water at base only",
            "Apply copper-based fungicide as alternative treatment"
        ],
        "prevention": [
            "Rotate crops every season",
            "Plant disease-resistant tomato varieties",
            "Ensure proper spacing between plants",
            "Remove plant debris after harvest"
        ]
    },
    "Tomato Late Blight": {
        "emoji": "🔴",
        "cause": "Water mold: Phytophthora infestans",
        "symptoms": "Dark watery spots on leaves and stems, white mold underneath",
        "treatment": [
            "Remove and destroy infected plants immediately",
            "Spray with Mancozeb or Metalaxyl every 5-7 days",
            "Do not compost infected plants — burn or bury them",
            "Apply treatment more frequently during wet weather"
        ],
        "prevention": [
            "Use drip irrigation — avoid overhead watering",
            "Plant certified disease-free seeds",
            "Ensure good drainage in the farm",
            "Monitor crops during rainy season"
        ]
    },
    "Tomato Healthy": {
        "emoji": "🟢",
        "cause": "No disease detected",
        "symptoms": "Plant appears completely healthy",
        "treatment": [
            "No treatment needed",
            "Continue regular watering and fertilization",
            "Monitor regularly for early signs of disease"
        ],
        "prevention": [
            "Maintain good soil nutrition",
            "Water consistently — avoid overwatering",
            "Keep farm clean and free of weeds",
            "Inspect plants regularly"
        ]
    }
}

# ── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detection | FUTB",
    page_icon="🌿",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f7f0; }
    .title {
        background: linear-gradient(135deg, #1a5276, #27ae60);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 15px 0;
    }
    .disease-box {
        background: #fdecea;
        border-left: 6px solid #e74c3c;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .healthy-box {
        background: #eafaf1;
        border-left: 6px solid #27ae60;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 30px;
        padding: 15px;
        border-top: 1px solid #ddd;
    }
    .confidence-bar {
        background: #eee;
        border-radius: 10px;
        padding: 3px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────
st.markdown("""
<div class="title">
    <h1>🌿 Plant Disease Detection System</h1>
    <p style="font-size:16px; margin:0;">Federal University of Technology Babura</p>
    <p style="font-size:14px; margin:5px 0 0 0; opacity:0.85;">
        AI-Powered Early Detection for Healthier Crops
    </p>
</div>
""", unsafe_allow_html=True)

# ── How to Use ─────────────────────────────────────
with st.expander("ℹ️ How to use this system"):
    st.write("1. 📸 Take a clear photo of a tomato plant leaf")
    st.write("2. 📤 Upload the photo using the button below")
    st.write("3. 🤖 Click 'Detect Disease' to analyse")
    st.write("4. 📋 Read the diagnosis and treatment advice")

st.markdown("---")

# ── Upload ─────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload a tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

    st.markdown("---")

    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("🤖 AI is analysing your leaf..."):
            img = image.resize((224, 224))
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            result = classes[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        info = treatments[result]

        # Result
        if result == "Tomato Healthy":
            st.markdown(f"""
            <div class="healthy-box">
                <h2>{info['emoji']} Result: {result}</h2>
                <p>Your plant appears to be healthy! Keep up the good work.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="disease-box">
                <h2>{info['emoji']} Disease Detected: {result}</h2>
                <p>Please follow the treatment recommendations below immediately.</p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence
        st.markdown(f"""
        <div class="info-card">
            <h4>📊 Confidence Score</h4>
            <h2 style="color:#1a5276; text-align:center;">{confidence:.2f}%</h2>
            <p style="text-align:center; color:#888;">
                The AI is {confidence:.1f}% confident in this diagnosis
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Disease Info
        st.markdown(f"""
        <div class="info-card">
            <h4>🔬 Disease Information</h4>
            <p><b>Cause:</b> {info['cause']}</p>
            <p><b>Symptoms:</b> {info['symptoms']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Treatment
        st.markdown('<div class="info-card"><h4>💊 Recommended Treatment</h4>', 
                   unsafe_allow_html=True)
        for t in info['treatment']:
            st.write(f"✅ {t}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Prevention
        st.markdown('<div class="info-card"><h4>🛡️ Prevention Tips</h4>', 
                   unsafe_allow_html=True)
        for p in info['prevention']:
            st.write(f"🔹 {p}")
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    Developed by <b>Yusuf Gambo</b> | 
    Matric No: SIT/CSC/23/0005 |
    FUTB Computer Science | 2024/2025<br>
    Supervised by <b>Dr. Khalid Haruna</b>
</div>
""", unsafe_allow_html=True)
