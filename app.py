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

diseases_db = {
    "tomato early blight": {
        "name": "Tomato Early Blight",
        "emoji": "🟠",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Brown spots with yellow rings on lower leaves",
        "treatment": [
            "Remove and destroy all infected leaves immediately",
            "Spray with fungicide containing Chlorothalonil every 7 days",
            "Avoid watering leaves directly — water at base only",
            "Apply copper-based fungicide as alternative"
        ],
        "prevention": [
            "Rotate crops every season",
            "Plant disease-resistant tomato varieties",
            "Ensure proper spacing between plants",
            "Remove plant debris after harvest"
        ]
    },
    "tomato late blight": {
        "name": "Tomato Late Blight",
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
    "tomato healthy": {
        "name": "Tomato Healthy",
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
    },
    "potato early blight": {
        "name": "Potato Early Blight",
        "emoji": "🟠",
        "cause": "Fungus: Alternaria solani",
        "symptoms": "Dark brown spots with concentric rings on older leaves",
        "treatment": [
            "Remove infected leaves immediately",
            "Apply fungicide containing Chlorothalonil or Mancozeb",
            "Spray every 7-10 days during growing season",
            "Avoid overhead irrigation"
        ],
        "prevention": [
            "Use certified disease-free seed potatoes",
            "Rotate crops every 2-3 years",
            "Maintain proper plant nutrition",
            "Remove crop debris after harvest"
        ]
    },
    "potato late blight": {
        "name": "Potato Late Blight",
        "emoji": "🔴",
        "cause": "Water mold: Phytophthora infestans",
        "symptoms": "Water-soaked spots on leaves turning brown, white mold on undersides",
        "treatment": [
            "Apply Metalaxyl or Cymoxanil fungicide immediately",
            "Remove and destroy all infected plant material",
            "Spray preventively before disease appears in wet weather",
            "Harvest tubers early if disease is severe"
        ],
        "prevention": [
            "Plant resistant potato varieties",
            "Avoid planting in poorly drained fields",
            "Monitor weather forecasts for blight conditions",
            "Use certified disease-free seed potatoes"
        ]
    },
    "corn common rust": {
        "name": "Corn Common Rust",
        "emoji": "🟠",
        "cause": "Fungus: Puccinia sorghi",
        "symptoms": "Small golden-brown pustules scattered on both leaf surfaces",
        "treatment": [
            "Apply fungicide containing Propiconazole or Azoxystrobin",
            "Spray at first sign of disease",
            "Repeat application every 14 days if needed",
            "Remove heavily infected plants"
        ],
        "prevention": [
            "Plant rust-resistant corn varieties",
            "Plant early to avoid peak rust season",
            "Monitor fields regularly",
            "Maintain good crop nutrition"
        ]
    },
    "cassava mosaic": {
        "name": "Cassava Mosaic Disease",
        "emoji": "🔴",
        "cause": "Virus transmitted by whitefly Bemisia tabaci",
        "symptoms": "Mosaic pattern of yellow and green on leaves, leaf distortion and stunting",
        "treatment": [
            "Remove and destroy infected plants immediately",
            "Control whitefly population with insecticide",
            "Use mineral oil spray to reduce virus spread",
            "There is no cure — prevention is key"
        ],
        "prevention": [
            "Plant certified virus-free cassava cuttings",
            "Use mosaic-resistant cassava varieties",
            "Control whitefly with neem-based insecticide",
            "Inspect new planting material before use"
        ]
    },
    "rice blast": {
        "name": "Rice Blast Disease",
        "emoji": "🔴",
        "cause": "Fungus: Magnaporthe oryzae",
        "symptoms": "Diamond-shaped lesions with grey centers and brown borders on leaves",
        "treatment": [
            "Apply fungicide containing Tricyclazole or Isoprothiolane",
            "Spray at booting stage and repeat after 10 days",
            "Drain fields periodically to reduce humidity",
            "Remove infected plant debris"
        ],
        "prevention": [
            "Plant blast-resistant rice varieties",
            "Avoid excessive nitrogen fertilization",
            "Maintain proper water management",
            "Use certified disease-free seeds"
        ]
    },
    "groundnut leaf spot": {
        "name": "Groundnut Leaf Spot",
        "emoji": "🟠",
        "cause": "Fungus: Cercospora arachidicola",
        "symptoms": "Dark brown circular spots on leaves, yellow halo around spots",
        "treatment": [
            "Spray with Chlorothalonil or Mancozeb fungicide",
            "Apply every 14 days from 30 days after planting",
            "Remove and destroy infected leaves",
            "Avoid overhead irrigation"
        ],
        "prevention": [
            "Use certified disease-free groundnut seeds",
            "Rotate crops with non-legume crops",
            "Remove crop debris after harvest",
            "Plant resistant groundnut varieties"
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

# ── TABS ───────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Upload Leaf Photo", "🔍 Search Disease by Name"])

# ════════════════════════════════════════════════
# TAB 1 — AI Photo Detection
# ════════════════════════════════════════════════
with tab1:
    st.markdown("### 📸 AI Disease Detection")
    st.write("Upload a tomato leaf photo and let AI detect the disease!")

    with st.expander("ℹ️ How to use"):
        st.write("1. 📸 Take a clear photo of a tomato plant leaf")
        st.write("2. 📤 Upload the photo below")
        st.write("3. 🔍 Click Detect Disease button")
        st.write("4. 📋 Read diagnosis and treatment advice")

    uploaded_file = st.file_uploader(
        "Upload a tomato leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Leaf", use_column_width=True)

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

            info = diseases_db[result.lower()]

            if result == "Tomato Healthy":
                st.markdown(f"""
                <div class="healthy-box">
                    <h2>{info['emoji']} Result: {result}</h2>
                    <p>Your plant appears healthy! Keep up the good work.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="disease-box">
                    <h2>{info['emoji']} Disease Detected: {result}</h2>
                    <p>Please follow treatment recommendations below immediately.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card">
                <h4>📊 Confidence Score</h4>
                <h2 style="color:#1a5276; text-align:center;">{confidence:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card">
                <h4>🔬 Disease Information</h4>
                <p><b>Cause:</b> {info['cause']}</p>
                <p><b>Symptoms:</b> {info['symptoms']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="info-card"><h4>💊 Treatment</h4>',
                       unsafe_allow_html=True)
            for t in info['treatment']:
                st.write(f"✅ {t}")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="info-card"><h4>🛡️ Prevention</h4>',
                       unsafe_allow_html=True)
            for p in info['prevention']:
                st.write(f"🔹 {p}")
            st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 — Search by Name
# ════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔍 Search Disease Information by Name")
    st.write("Type the name of a crop or disease to get information!")

    search_query = st.text_input(
        "Type crop or disease name:",
        placeholder="e.g. tomato, potato, cassava, rice, corn..."
    )

    if search_query:
        query = search_query.lower().strip()
        found = []

        for key, info in diseases_db.items():
            if query in key or query in info['name'].lower():
                found.append(info)

        if found:
            st.success(f"✅ Found {len(found)} result(s) for '{search_query}'")

            for info in found:
                with st.expander(f"{info['emoji']} {info['name']}"):
                    st.markdown(f"""
                    <div class="info-card">
                        <p><b>🔬 Cause:</b> {info['cause']}</p>
                        <p><b>👁️ Symptoms:</b> {info['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**💊 Treatment:**")
                    for t in info['treatment']:
                        st.write(f"✅ {t}")

                    st.markdown("**🛡️ Prevention:**")
                    for p in info['prevention']:
                        st.write(f"🔹 {p}")
        else:
            st.warning(f"❌ No results found for '{search_query}'")
            st.write("Try searching for: **tomato, potato, cassava, rice, corn, groundnut**")

# ── Footer ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    Developed by <b>Yusuf Gambo</b> |
    Matric No: SIT/CSC/23/0005 |
    FUTB Computer Science | 2024/2025<br>
    Supervised by <b>Dr. Khalid Haruna</b>
</div>
""", unsafe_allow_html=True)
