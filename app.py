import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection | FUTB",
    page_icon="🌿",
    layout="centered"
)

model_loaded = False
try:
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path="plant_disease_model_38.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    model_loaded = True
except:
    pass

ai_classes = [
    "Apple Black Rot", "Apple Cedar Rust", "Apple Healthy", "Apple Scab",
    "Blueberry Healthy", "Cherry Healthy", "Cherry Powdery Mildew",
    "Corn Cercospora Leaf Spot", "Corn Common Rust", "Corn Healthy",
    "Corn Northern Leaf Blight", "Grape Black Measles", "Grape Black Rot",
    "Grape Healthy", "Grape Leaf Blight", "Orange Citrus Greening",
    "Peach Bacterial Spot", "Peach Healthy", "Pepper Bacterial Spot",
    "Pepper Healthy", "Potato Early Blight", "Potato Healthy",
    "Potato Late Blight", "Raspberry Healthy", "Soybean Healthy",
    "Squash Powdery Mildew", "Strawberry Healthy", "Strawberry Leaf Scorch",
    "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Healthy",
    "Tomato Late Blight", "Tomato Leaf Mold", "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
    "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus"
]

diseases_db = {
    "tomato bacterial spot": {"name":"Tomato Bacterial Spot","crop":"Tomato","emoji":"🔴","cause":"Bacterium: Xanthomonas vesicatoria","symptoms":"Small dark water-soaked spots on leaves turning brown with yellow halo.","treatment":["Apply copper-based bactericide every 7 days","Remove infected plant parts immediately","Avoid working with plants when wet","Use streptomycin spray in severe cases"],"prevention":["Use certified disease-free seeds","Avoid overhead irrigation","Rotate crops every season","Disinfect tools regularly"]},
    "tomato early blight": {"name":"Tomato Early Blight","crop":"Tomato","emoji":"🟠","cause":"Fungus: Alternaria solani","symptoms":"Brown spots with yellow rings on lower leaves. Leaves turn yellow and drop.","treatment":["Remove infected leaves immediately","Spray Chlorothalonil fungicide every 7 days","Apply copper-based fungicide as alternative","Water at base only"],"prevention":["Rotate crops every season","Plant resistant varieties","Proper spacing for air circulation","Remove plant debris after harvest"]},
    "tomato late blight": {"name":"Tomato Late Blight","crop":"Tomato","emoji":"🔴","cause":"Water mold: Phytophthora infestans","symptoms":"Dark watery spots on leaves and stems. White mold visible underneath.","treatment":["Remove and destroy infected plants immediately","Spray Mancozeb or Metalaxyl every 5-7 days","Do not compost — burn infected plants","Increase spray frequency in wet weather"],"prevention":["Use drip irrigation only","Plant certified disease-free seeds","Ensure good drainage","Monitor during rainy season"]},
    "tomato leaf mold": {"name":"Tomato Leaf Mold","crop":"Tomato","emoji":"🟡","cause":"Fungus: Passalora fulva","symptoms":"Yellow spots on upper leaf surface with olive-green mold on underside.","treatment":["Apply Chlorothalonil or Mancozeb","Improve air circulation","Remove infected leaves","Reduce humidity around plants"],"prevention":["Plant resistant varieties","Space plants properly","Avoid overhead watering","Maintain low humidity"]},
    "tomato mosaic virus": {"name":"Tomato Mosaic Virus","crop":"Tomato","emoji":"🔴","cause":"Virus: Tomato Mosaic Virus (ToMV)","symptoms":"Mosaic pattern of light and dark green on leaves, stunted growth.","treatment":["Remove and destroy infected plants","Control aphid vectors with insecticide","Disinfect tools with bleach solution","Wash hands before handling plants"],"prevention":["Use virus-free certified seeds","Control insect vectors","Remove infected plants immediately","Avoid tobacco near plants"]},
    "tomato septoria leaf spot": {"name":"Tomato Septoria Leaf Spot","crop":"Tomato","emoji":"🟠","cause":"Fungus: Septoria lycopersici","symptoms":"Small circular spots with dark borders and light grey centers on leaves.","treatment":["Apply Chlorothalonil or copper fungicide","Remove infected lower leaves","Spray every 7-10 days","Avoid wetting foliage when watering"],"prevention":["Rotate crops for 2 years","Use mulch to prevent soil splash","Space plants for good airflow","Remove crop debris"]},
    "tomato spider mites": {"name":"Tomato Spider Mites","crop":"Tomato","emoji":"🟡","cause":"Pest: Tetranychus urticae","symptoms":"Yellow stippling on leaves, fine webbing on underside. Leaves turn bronze.","treatment":["Apply miticide or insecticidal soap","Spray neem oil every 5 days","Increase humidity around plants","Remove heavily infested leaves"],"prevention":["Monitor regularly","Avoid water stress","Remove infested leaves","Use reflective mulch"]},
    "tomato target spot": {"name":"Tomato Target Spot","crop":"Tomato","emoji":"🟠","cause":"Fungus: Corynespora cassiicola","symptoms":"Brown spots with concentric rings resembling a target on leaves and fruits.","treatment":["Apply Chlorothalonil or Mancozeb fungicide","Remove infected plant material","Spray every 7-14 days","Improve air circulation"],"prevention":["Use disease-free transplants","Rotate crops regularly","Avoid overhead irrigation","Remove crop debris"]},
    "tomato yellow leaf curl virus": {"name":"Tomato Yellow Leaf Curl Virus","crop":"Tomato","emoji":"🔴","cause":"Virus transmitted by whitefly Bemisia tabaci","symptoms":"Upward curling and yellowing of leaves, stunted growth, reduced fruit.","treatment":["Remove and destroy infected plants","Apply insecticide to control whiteflies","Use yellow sticky traps","No chemical cure for the virus"],"prevention":["Plant resistant varieties","Use insect-proof screens","Control whitefly with neem oil","Use reflective mulch"]},
    "tomato healthy": {"name":"Tomato Healthy","crop":"Tomato","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed","Continue regular care"],"prevention":["Maintain good soil nutrition","Monitor regularly"]},
    "potato early blight": {"name":"Potato Early Blight","crop":"Potato","emoji":"🟠","cause":"Fungus: Alternaria solani","symptoms":"Dark brown spots with concentric rings on older leaves.","treatment":["Apply Chlorothalonil or Mancozeb","Remove infected leaves","Spray every 7-10 days","Avoid overhead irrigation"],"prevention":["Use certified seed potatoes","Rotate crops every 2-3 years","Maintain proper nutrition","Remove crop debris"]},
    "potato late blight": {"name":"Potato Late Blight","crop":"Potato","emoji":"🔴","cause":"Water mold: Phytophthora infestans","symptoms":"Water-soaked spots turning brown. White mold on undersides. Tubers rot.","treatment":["Apply Metalaxyl or Cymoxanil immediately","Remove all infected material","Harvest early if severe","Spray preventively in wet weather"],"prevention":["Plant resistant varieties","Avoid poorly drained fields","Use certified seed potatoes","Monitor weather forecasts"]},
    "potato healthy": {"name":"Potato Healthy","crop":"Potato","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "corn common rust": {"name":"Corn Common Rust","crop":"Corn/Maize","emoji":"🟠","cause":"Fungus: Puccinia sorghi","symptoms":"Small golden-brown pustules scattered on both leaf surfaces.","treatment":["Apply Propiconazole or Azoxystrobin","Spray at first sign of disease","Repeat application every 14 days","Remove heavily infected plants"],"prevention":["Plant rust-resistant corn varieties","Plant early to avoid peak rust season","Monitor fields regularly","Maintain good crop nutrition"]},
    "corn northern leaf blight": {"name":"Corn Northern Leaf Blight","crop":"Corn/Maize","emoji":"🟠","cause":"Fungus: Exserohilum turcicum","symptoms":"Long tan-grey cigar-shaped lesions on leaves.","treatment":["Apply Propiconazole or Tebuconazole","Spray at tasseling stage","Remove infected debris","Avoid dense planting"],"prevention":["Plant resistant hybrids","Rotate crops","Till soil to bury debris","Avoid excessive nitrogen"]},
    "corn cercospora leaf spot": {"name":"Corn Cercospora Leaf Spot","crop":"Corn/Maize","emoji":"🟡","cause":"Fungus: Cercospora zeae-maydis","symptoms":"Rectangular grey to tan lesions running between leaf veins.","treatment":["Apply Strobilurin or Triazole fungicide","Spray at early disease onset","Improve field drainage","Remove infected residue"],"prevention":["Plant resistant varieties","Rotate crops","Reduce plant density","Avoid minimum tillage"]},
    "corn healthy": {"name":"Corn Healthy","crop":"Corn/Maize","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "rice blast": {"name":"Rice Blast Disease","crop":"Rice","emoji":"🔴","cause":"Fungus: Magnaporthe oryzae","symptoms":"Diamond-shaped lesions with grey centers and brown borders on leaves.","treatment":["Apply Tricyclazole or Isoprothiolane","Spray at booting stage and repeat after 10 days","Drain fields periodically","Remove infected plant debris"],"prevention":["Plant blast-resistant rice varieties","Avoid excessive nitrogen fertilization","Maintain proper water management","Use certified disease-free seeds"]},
    "rice brown spot": {"name":"Rice Brown Spot","crop":"Rice","emoji":"🟠","cause":"Fungus: Cochliobolus miyabeanus","symptoms":"Oval brown spots with yellow halo on leaves.","treatment":["Apply Mancozeb or Iprodione","Spray at tillering stage","Improve soil fertility","Remove infected debris"],"prevention":["Use certified seeds","Maintain proper nutrition","Avoid water stress","Rotate crops"]},
    "rice bacterial blight": {"name":"Rice Bacterial Blight","crop":"Rice","emoji":"🔴","cause":"Bacterium: Xanthomonas oryzae","symptoms":"Water-soaked lesions on leaf margins turning yellow then white.","treatment":["Apply copper-based bactericide","Drain fields and keep dry","Remove infected plants","Avoid excessive nitrogen"],"prevention":["Plant resistant varieties","Use certified seeds","Avoid flood irrigation","Maintain field hygiene"]},
    "rice sheath blight": {"name":"Rice Sheath Blight","crop":"Rice","emoji":"🟠","cause":"Fungus: Rhizoctonia solani","symptoms":"Oval lesions on leaf sheaths near water line with brown borders.","treatment":["Apply Propiconazole or Hexaconazole","Spray at early tillering","Drain field to reduce humidity","Remove infected stubble"],"prevention":["Reduce plant density","Avoid excessive nitrogen","Use resistant varieties","Rotate crops"]},
    "rice tungro": {"name":"Rice Tungro Disease","crop":"Rice","emoji":"🔴","cause":"Virus transmitted by green leafhopper","symptoms":"Yellow-orange discoloration of leaves, stunted growth.","treatment":["Control leafhopper with insecticide","Remove infected plants","Use systemic insecticide"],"prevention":["Plant tungro-resistant varieties","Control leafhopper","Synchronize planting dates","Remove infected plants early"]},
    "rice healthy": {"name":"Rice Healthy","crop":"Rice","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "cassava mosaic": {"name":"Cassava Mosaic Disease","crop":"Cassava","emoji":"🔴","cause":"Virus transmitted by whitefly Bemisia tabaci","symptoms":"Mosaic pattern of yellow and green on leaves, distortion, stunted growth.","treatment":["Remove and destroy infected plants","Control whitefly with insecticide","Use mineral oil spray","No chemical cure"],"prevention":["Plant certified virus-free cuttings","Use mosaic-resistant varieties","Control whitefly with neem","Inspect planting material"]},
    "cassava brown streak": {"name":"Cassava Brown Streak Disease","crop":"Cassava","emoji":"🔴","cause":"Virus: Cassava Brown Streak Virus","symptoms":"Yellow patches on leaves, brown streaks on stems, brown patches in tubers.","treatment":["Remove all infected plants","Control whitefly vectors","No chemical treatment","Replace with resistant varieties"],"prevention":["Use CBSD-resistant varieties","Plant certified cuttings","Control whitefly","Avoid moving material from infected areas"]},
    "cassava bacterial blight": {"name":"Cassava Bacterial Blight","crop":"Cassava","emoji":"🟠","cause":"Bacterium: Xanthomonas axonopodis","symptoms":"Angular water-soaked leaf spots, wilting, stem cankers.","treatment":["Apply copper-based bactericide","Remove infected parts","Disinfect cutting tools","Destroy severely infected plants"],"prevention":["Use disease-free planting material","Disinfect tools","Plant resistant varieties","Avoid infected soil"]},
    "cassava healthy": {"name":"Cassava Healthy","crop":"Cassava","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "groundnut early leaf spot": {"name":"Groundnut Early Leaf Spot","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Cercospora arachidicola","symptoms":"Dark brown circular spots on upper leaf surface with yellow halo.","treatment":["Spray Chlorothalonil or Mancozeb","Apply every 14 days","Remove infected leaves","Avoid overhead irrigation"],"prevention":["Use certified seeds","Rotate crops","Remove crop debris","Plant resistant varieties"]},
    "groundnut late leaf spot": {"name":"Groundnut Late Leaf Spot","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Cercosporidium personatum","symptoms":"Dark brown to black spots on lower leaf surface.","treatment":["Apply Tebuconazole or Propiconazole","Spray every 14 days","Remove infected material","Improve air circulation"],"prevention":["Rotate crops","Use resistant varieties","Remove crop debris","Avoid dense planting"]},
    "groundnut rosette": {"name":"Groundnut Rosette Disease","crop":"Groundnut","emoji":"🔴","cause":"Virus transmitted by aphid Aphis craccivora","symptoms":"Stunted plants with small mottled leaves, chlorotic rosette pattern.","treatment":["Control aphid with insecticide","Remove infected plants","Apply mineral oil to reduce spread"],"prevention":["Plant early","Use resistant varieties","Control aphid with neem","Plant barrier crops"]},
    "groundnut rust": {"name":"Groundnut Rust","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Puccinia arachidis","symptoms":"Orange-brown pustules on lower leaf surfaces, yellowing and defoliation.","treatment":["Apply Mancozeb or Propiconazole","Spray every 14 days","Remove infected leaves"],"prevention":["Plant resistant varieties","Rotate crops","Remove crop debris","Monitor fields regularly"]},
    "pepper bacterial spot": {"name":"Pepper Bacterial Spot","crop":"Pepper","emoji":"🟠","cause":"Bacterium: Xanthomonas campestris","symptoms":"Small water-soaked spots on leaves turning brown with yellow halo.","treatment":["Apply copper bactericide every 7 days","Remove infected parts","Avoid wet plants"],"prevention":["Use certified seeds","Avoid overhead irrigation","Rotate crops","Disinfect tools"]},
    "pepper healthy": {"name":"Pepper Healthy","crop":"Pepper","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "onion purple blotch": {"name":"Onion Purple Blotch","crop":"Onion","emoji":"🟠","cause":"Fungus: Alternaria porri","symptoms":"Small white lesions with purple centers on leaves enlarging to kill leaves.","treatment":["Apply Mancozeb or Iprodione","Spray every 7-10 days","Remove infected leaves","Improve air circulation"],"prevention":["Use certified disease-free sets","Avoid overhead irrigation","Rotate crops","Remove crop debris"]},
    "apple scab": {"name":"Apple Scab","crop":"Apple","emoji":"🟠","cause":"Fungus: Venturia inaequalis","symptoms":"Olive-green to brown scab-like lesions on leaves and fruits.","treatment":["Apply Myclobutanil or Captan fungicide","Spray from bud break","Remove infected leaves","Prune for air circulation"],"prevention":["Plant resistant varieties","Remove fallen leaves","Apply dormant sprays","Prune for good airflow"]},
    "apple black rot": {"name":"Apple Black Rot","crop":"Apple","emoji":"🔴","cause":"Fungus: Botryosphaeria obtusa","symptoms":"Brown circular lesions on leaves, black rotting of fruits, cankers on branches.","treatment":["Apply Captan or Thiophanate-methyl","Remove infected fruits and branches","Spray every 7-10 days","Prune cankers from branches"],"prevention":["Remove mummified fruits","Prune infected branches","Maintain tree vigor","Apply dormant copper spray"]},
    "grape black rot": {"name":"Grape Black Rot","crop":"Grape","emoji":"🔴","cause":"Fungus: Guignardia bidwellii","symptoms":"Brown circular lesions on leaves, shriveled black mummified berries.","treatment":["Apply Myclobutanil or Mancozeb","Spray from bud break","Remove infected berries","Repeat every 7-14 days"],"prevention":["Remove mummified berries","Prune for good air circulation","Apply early season sprays","Remove infected plant material"]},
}

# ── Beautiful CSS ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.main { background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f1 100%); }

.hero {
    background: linear-gradient(135deg, #1a5276 0%, #1e8449 50%, #27ae60 100%);
    color: white;
    padding: 40px 30px;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(26,82,118,0.3);
    animation: fadeIn 1s ease;
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.hero p { opacity: 0.9; margin: 5px 0 0 0; }

.stat-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 4px solid #27ae60;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-3px); }
.stat-num { font-size: 2rem; font-weight: 700; color: #1a5276; }
.stat-label { color: #666; font-size: 0.85rem; }

.crop-badge {
    background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
    border: 1px solid #a5d6a7;
    border-radius: 20px;
    padding: 8px 15px;
    display: inline-block;
    margin: 4px;
    font-size: 0.85rem;
    color: #1b5e20;
    font-weight: 500;
}

.upload-box {
    border: 2px dashed #27ae60;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
    background: white;
    margin: 15px 0;
}

.disease-result {
    background: linear-gradient(135deg, #fdecea, #fce4e4);
    border-left: 6px solid #e74c3c;
    border-radius: 12px;
    padding: 20px 25px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(231,76,60,0.15);
}
.disease-result h2 { color: #c0392b; margin: 0 0 5px 0; }

.healthy-result {
    background: linear-gradient(135deg, #eafaf1, #d5f5e3);
    border-left: 6px solid #27ae60;
    border-radius: 12px;
    padding: 20px 25px;
    margin: 15px 0;
    box-shadow: 0 4px 15px rgba(39,174,96,0.15);
}
.healthy-result h2 { color: #1e8449; margin: 0 0 5px 0; }

.confidence-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    margin: 15px 0;
}
.confidence-num {
    font-size: 3rem;
    font-weight: 700;
    color: #1a5276;
    line-height: 1;
}

.info-section {
    background: white;
    border-radius: 15px;
    padding: 20px 25px;
    margin: 12px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    border-left: 4px solid #3498db;
}
.info-section h4 {
    color: #1a5276;
    margin: 0 0 12px 0;
    font-size: 1.1rem;
}

.treatment-item {
    background: #e8f5e9;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    color: #1b5e20;
    font-size: 0.9rem;
}
.prevention-item {
    background: #e3f2fd;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    color: #1565c0;
    font-size: 0.9rem;
}

.search-result-card {
    background: white;
    border-radius: 15px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    border-top: 3px solid #27ae60;
}

.how-to-step {
    background: white;
    border-radius: 10px;
    padding: 12px 15px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex;
    align-items: center;
    gap: 10px;
}

.footer {
    background: linear-gradient(135deg, #1a5276, #1e8449);
    color: white;
    text-align: center;
    padding: 20px;
    border-radius: 15px;
    margin-top: 30px;
    font-size: 13px;
}
.footer a { color: #a9dfbf; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ──────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detection System</h1>
    <p style="font-size:1rem;">Federal University of Technology Babura</p>
    <p style="font-size:0.85rem; opacity:0.8;">
        🤖 AI-Powered | 🆓 Free | 📱 Works on Any Device
    </p>
</div>
""", unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-card"><div class="stat-num">14+</div><div class="stat-label">🌾 Crops</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-card"><div class="stat-num">47+</div><div class="stat-label">🦠 Diseases</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-card"><div class="stat-num">95%</div><div class="stat-label">🎯 Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-card"><div class="stat-num">🆓</div><div class="stat-label">💰 Free</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📸 AI Photo Detection",
    "🔍 Search by Name",
    "🌾 Browse by Crop"
])

# ════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════
with tab1:
    st.markdown("## 📸 AI Plant Disease Detection")
    st.write("Upload a clear photo of a plant leaf and our AI will instantly identify any disease!")

    # How to use
    with st.expander("📖 How to use — Click to expand"):
        st.markdown("""
        <div class="how-to-step">⬆️ <b>Step 1:</b> Upload a clear photo of a plant leaf below</div>
        <div class="how-to-step">🔍 <b>Step 2:</b> Click the "Detect Disease" button</div>
        <div class="how-to-step">📋 <b>Step 3:</b> Read the diagnosis and follow treatment advice</div>
        <div class="how-to-step">💡 <b>Tip:</b> Use a well-lit, close-up photo for best results</div>
        """, unsafe_allow_html=True)

    # Supported crops
    st.markdown("### ✅ Crops the AI Can Detect:")
    st.markdown("""
    <div style="background:white;padding:15px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:15px;">
    <span class="crop-badge">🍎 Apple</span>
    <span class="crop-badge">🫐 Blueberry</span>
    <span class="crop-badge">🍒 Cherry</span>
    <span class="crop-badge">🌽 Corn/Maize</span>
    <span class="crop-badge">🍇 Grape</span>
    <span class="crop-badge">🍊 Orange</span>
    <span class="crop-badge">🍑 Peach</span>
    <span class="crop-badge">🌶️ Pepper</span>
    <span class="crop-badge">🥔 Potato</span>
    <span class="crop-badge">🍓 Raspberry</span>
    <span class="crop-badge">🫘 Soybean</span>
    <span class="crop-badge">🎃 Squash</span>
    <span class="crop-badge">🍓 Strawberry</span>
    <span class="crop-badge">🍅 Tomato</span>
    </div>
    <p style="color:#e74c3c;font-size:0.85rem;">
    ⚠️ <b>For Rice, Cassava, Groundnut, Onion</b> — 
    use the <b>🔍 Search</b> or <b>🌾 Browse</b> tab!
    </p>
    """, unsafe_allow_html=True)

    # Upload
    uploaded_file = st.file_uploader(
        "📤 Choose a leaf image (JPG or PNG)",
        type=["jpg","jpeg","png"],
        help="Upload a clear photo of the plant leaf"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(image, caption="📷 Uploaded Leaf Image",
                    use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button(
            "🔍 DETECT DISEASE NOW",
            use_container_width=True,
            type="primary"
        )

        if detect_btn:
            if model_loaded:
                with st.spinner("🤖 AI is analysing your leaf... Please wait..."):
                    img = image.resize((224,224))
                    img = np.array(img, dtype=np.float32)/255.0
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_details[0]['index'], img)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    result = ai_classes[np.argmax(prediction)]
                    confidence = np.max(prediction)*100

                st.markdown("---")
                st.markdown("## 📊 Detection Results")

                if "Healthy" in result:
                    st.markdown(f"""
                    <div class="healthy-result">
                        <h2>🟢 {result}</h2>
                        <p>Great news! Your plant appears to be completely healthy.
                        Keep monitoring it regularly to maintain its health.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="disease-result">
                        <h2>⚠️ {result}</h2>
                        <p>Disease detected! Please follow the treatment 
                        recommendations below as soon as possible to prevent 
                        further spread.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Confidence
                st.markdown(f"""
                <div class="confidence-card">
                    <p style="color:#666;margin:0;">🎯 AI Confidence Score</p>
                    <div class="confidence-num">{confidence:.1f}%</div>
                    <p style="color:#888;font-size:0.85rem;margin:5px 0 0 0;">
                        The AI is {confidence:.1f}% confident in this diagnosis
                    </p>
                </div>
                """, unsafe_allow_html=True)

                info = diseases_db.get(result.lower(), None)
                if info:
                    # Disease Info
                    st.markdown(f"""
                    <div class="info-section">
                        <h4>🔬 Disease Information</h4>
                        <p><b>Disease Name:</b> {info['name']}</p>
                        <p><b>Affected Crop:</b> {info['crop']}</p>
                        <p><b>Cause:</b> {info['cause']}</p>
                        <p><b>Symptoms:</b> {info['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Treatment
                    st.markdown('<div class="info-section" style="border-left-color:#27ae60;"><h4>💊 Recommended Treatment</h4>', unsafe_allow_html=True)
                    for t in info["treatment"]:
                        st.markdown(f'<div class="treatment-item">✅ {t}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Prevention
                    st.markdown('<div class="info-section" style="border-left-color:#3498db;"><h4>🛡️ Prevention Tips</h4>', unsafe_allow_html=True)
                    for p in info["prevention"]:
                        st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.success("✅ Analysis complete! Follow the treatment steps above for best results.")
            else:
                st.error("❌ AI Model not loaded. Please check deployment configuration.")

# ════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Search Disease Information")
    st.write("Type any crop name or disease name to get complete information instantly!")

    st.markdown("""
    <div style="background:#e8f5e9;border-radius:10px;padding:12px 15px;margin-bottom:15px;">
    💡 <b>Try searching for:</b> tomato, rice, cassava, blight, rust, maize, groundnut, pepper, mosaic, mildew
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input(
        "🔎 Type crop or disease name here:",
        placeholder="e.g. tomato, rice blast, cassava mosaic, blight...",
    )

    if query:
        q = query.lower().strip()
        found = [
            info for key, info in diseases_db.items()
            if q in key
            or q in info["name"].lower()
            or q in info["crop"].lower()
            or q in info["symptoms"].lower()
            or q in info["cause"].lower()
        ]

        if found:
            st.success(f"✅ Found **{len(found)}** result(s) for **'{query}'**")
            for info in found:
                with st.expander(f"{info['emoji']} {info['name']} — {info['crop']}"):
                    st.markdown(f"""
                    <div class="info-section">
                        <h4>🔬 Disease Details</h4>
                        <p><b>🌾 Crop:</b> {info['crop']}</p>
                        <p><b>🧫 Cause:</b> {info['cause']}</p>
                        <p><b>👁️ Symptoms:</b> {info['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**💊 Treatment Steps:**")
                    for t in info["treatment"]:
                        st.markdown(f'<div class="treatment-item">✅ {t}</div>', unsafe_allow_html=True)

                    st.markdown("**🛡️ Prevention Tips:**")
                    for p in info["prevention"]:
                        st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)
        else:
            st.warning(f"❌ No results found for **'{query}'**")
            st.markdown("""
            <div style="background:white;border-radius:10px;padding:15px;margin-top:10px;">
            <b>Try searching for these crops:</b><br><br>
            🍅 tomato &nbsp;|&nbsp; 🥔 potato &nbsp;|&nbsp; 🌾 rice &nbsp;|&nbsp;
            🌿 cassava &nbsp;|&nbsp; 🌽 maize &nbsp;|&nbsp; 🥜 groundnut &nbsp;|&nbsp;
            🌶️ pepper &nbsp;|&nbsp; 🧅 onion &nbsp;|&nbsp; 🍎 apple &nbsp;|&nbsp; 🍇 grape
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════
with tab3:
    st.markdown("## 🌾 Browse Diseases by Crop")
    st.write("Select any crop to see all its diseases, symptoms, and treatment advice!")

    crops = sorted(set(info["crop"] for info in diseases_db.values()))

    crop_emojis = {
        "Apple": "🍎", "Cassava": "🌿", "Corn/Maize": "🌽",
        "Grape": "🍇", "Groundnut": "🥜", "Onion": "🧅",
        "Pepper": "🌶️", "Potato": "🥔", "Rice": "🌾",
        "Tomato": "🍅"
    }

    selected = st.selectbox(
        "🌾 Select a crop:",
        ["-- Select a crop to view diseases --"] + crops,
        help="Choose a crop to see all associated diseases"
    )

    if selected != "-- Select a crop to view diseases --":
        matches = [
            info for info in diseases_db.values()
            if info["crop"] == selected
        ]
        emoji = crop_emojis.get(selected, "🌿")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#27ae60);color:white;
        padding:15px 20px;border-radius:12px;margin:10px 0;">
            <h3 style="margin:0;">{emoji} {selected}</h3>
            <p style="margin:5px 0 0 0;opacity:0.9;">
                Found <b>{len(matches)} diseases</b> in database
            </p>
        </div>
        """, unsafe_allow_html=True)

        healthy = [m for m in matches if "Healthy" in m["name"]]
        diseases = [m for m in matches if "Healthy" not in m["name"]]

        if healthy:
            st.markdown("### ✅ Healthy State")
            for info in healthy:
                with st.expander(f"🟢 {info['name']}"):
                    st.success("This plant is in a healthy state with no disease detected.")
                    for p in info["prevention"]:
                        st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)

        if diseases:
            st.markdown(f"### ⚠️ {len(diseases)} Disease(s) Found")
            for info in diseases:
                with st.expander(f"{info['emoji']} {info['name']}"):
                    st.markdown(f"""
                    <div class="info-section">
                        <h4>🔬 Disease Information</h4>
                        <p><b>Cause:</b> {info['cause']}</p>
                        <p><b>Symptoms:</b> {info['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**💊 Treatment:**")
                        for t in info["treatment"]:
                            st.markdown(f'<div class="treatment-item">✅ {t}</div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown("**🛡️ Prevention:**")
                        for p in info["prevention"]:
                            st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p style="margin:0;font-size:1rem;">🌿 Plant Disease Detection System</p>
    <p style="margin:5px 0;opacity:0.9;">
        Developed by <b>Yusuf Gambo</b> | Matric: SIT/CSC/23/0005<br>
        B.Sc Computer Science | FUTB | 2024/2025<br>
        Supervised by <b>Dr. Khalid Haruna</b>
    </p>
    <a href="https://futb-plant-disease.streamlit.app" 
       style="color:#a9dfbf;">
       🌐 futb-plant-disease.streamlit.app
    </a>
</div>
""", unsafe_allow_html=True)
