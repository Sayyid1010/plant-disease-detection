import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Plant Disease Detection | FUTB",
    page_icon="🌿",
    layout="centered"
)

# ── Load Disease Model ───────────────────────────
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

# ── Load Leaf Validator Model ────────────────────
validator_loaded = False
try:
    from ai_edge_litert.interpreter import Interpreter as Interpreter2
    validator = Interpreter2(model_path="leaf_validator.tflite")
    validator.allocate_tensors()
    val_input = validator.get_input_details()
    val_output = validator.get_output_details()
    validator_loaded = True
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
    "tomato early blight": {"name":"Tomato Early Blight","crop":"Tomato","emoji":"🟠","cause":"Fungus: Alternaria solani","symptoms":"Brown spots with yellow rings on lower leaves.","treatment":["Remove infected leaves immediately","Spray Chlorothalonil fungicide every 7 days","Apply copper-based fungicide as alternative","Water at base only"],"prevention":["Rotate crops every season","Plant resistant varieties","Proper spacing for air circulation","Remove plant debris after harvest"]},
    "tomato late blight": {"name":"Tomato Late Blight","crop":"Tomato","emoji":"🔴","cause":"Water mold: Phytophthora infestans","symptoms":"Dark watery spots on leaves and stems. White mold visible underneath.","treatment":["Remove and destroy infected plants immediately","Spray Mancozeb or Metalaxyl every 5-7 days","Do not compost — burn infected plants","Increase spray in wet weather"],"prevention":["Use drip irrigation only","Plant certified disease-free seeds","Ensure good drainage","Monitor during rainy season"]},
    "tomato leaf mold": {"name":"Tomato Leaf Mold","crop":"Tomato","emoji":"🟡","cause":"Fungus: Passalora fulva","symptoms":"Yellow spots on upper leaf surface with olive-green mold on underside.","treatment":["Apply Chlorothalonil or Mancozeb","Improve air circulation","Remove infected leaves","Reduce humidity"],"prevention":["Plant resistant varieties","Space plants properly","Avoid overhead watering","Maintain low humidity"]},
    "tomato mosaic virus": {"name":"Tomato Mosaic Virus","crop":"Tomato","emoji":"🔴","cause":"Virus: Tomato Mosaic Virus","symptoms":"Mosaic pattern on leaves, stunted growth, distorted fruits.","treatment":["Remove infected plants","Control aphid vectors","Disinfect tools with bleach","Wash hands before handling"],"prevention":["Use virus-free seeds","Control insect vectors","Remove infected plants immediately","Avoid tobacco near plants"]},
    "tomato septoria leaf spot": {"name":"Tomato Septoria Leaf Spot","crop":"Tomato","emoji":"🟠","cause":"Fungus: Septoria lycopersici","symptoms":"Small circular spots with dark borders and grey centers.","treatment":["Apply Chlorothalonil or copper fungicide","Remove infected lower leaves","Spray every 7-10 days","Avoid wetting foliage"],"prevention":["Rotate crops for 2 years","Use mulch","Space plants for airflow","Remove crop debris"]},
    "tomato spider mites": {"name":"Tomato Spider Mites","crop":"Tomato","emoji":"🟡","cause":"Pest: Tetranychus urticae","symptoms":"Yellow stippling on leaves, fine webbing. Leaves turn bronze.","treatment":["Apply miticide or insecticidal soap","Spray neem oil every 5 days","Increase humidity","Remove infested leaves"],"prevention":["Monitor regularly","Avoid water stress","Remove infested leaves","Use reflective mulch"]},
    "tomato target spot": {"name":"Tomato Target Spot","crop":"Tomato","emoji":"🟠","cause":"Fungus: Corynespora cassiicola","symptoms":"Brown spots with concentric rings on leaves and fruits.","treatment":["Apply Chlorothalonil or Mancozeb","Remove infected material","Spray every 7-14 days","Improve air circulation"],"prevention":["Use disease-free transplants","Rotate crops","Avoid overhead irrigation","Remove crop debris"]},
    "tomato yellow leaf curl virus": {"name":"Tomato Yellow Leaf Curl Virus","crop":"Tomato","emoji":"🔴","cause":"Virus transmitted by whitefly","symptoms":"Upward curling and yellowing of leaves, stunted growth.","treatment":["Remove infected plants","Apply insecticide for whiteflies","Use yellow sticky traps","No chemical cure"],"prevention":["Plant resistant varieties","Use insect-proof screens","Control whitefly with neem","Use reflective mulch"]},
    "tomato healthy": {"name":"Tomato Healthy","crop":"Tomato","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed","Continue regular care"],"prevention":["Maintain good soil nutrition","Monitor regularly"]},
    "potato early blight": {"name":"Potato Early Blight","crop":"Potato","emoji":"🟠","cause":"Fungus: Alternaria solani","symptoms":"Dark brown spots with concentric rings on older leaves.","treatment":["Apply Chlorothalonil or Mancozeb","Remove infected leaves","Spray every 7-10 days","Avoid overhead irrigation"],"prevention":["Use certified seed potatoes","Rotate crops","Maintain proper nutrition","Remove crop debris"]},
    "potato late blight": {"name":"Potato Late Blight","crop":"Potato","emoji":"🔴","cause":"Water mold: Phytophthora infestans","symptoms":"Water-soaked spots turning brown. White mold on undersides.","treatment":["Apply Metalaxyl immediately","Remove all infected material","Harvest early if severe","Spray preventively in wet weather"],"prevention":["Plant resistant varieties","Avoid poorly drained fields","Use certified seed potatoes","Monitor weather"]},
    "potato healthy": {"name":"Potato Healthy","crop":"Potato","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "corn common rust": {"name":"Corn Common Rust","crop":"Corn/Maize","emoji":"🟠","cause":"Fungus: Puccinia sorghi","symptoms":"Small golden-brown pustules on both leaf surfaces.","treatment":["Apply Propiconazole or Azoxystrobin","Spray at first sign","Repeat every 14 days","Remove infected plants"],"prevention":["Plant rust-resistant varieties","Plant early","Monitor regularly","Maintain good nutrition"]},
    "corn northern leaf blight": {"name":"Corn Northern Leaf Blight","crop":"Corn/Maize","emoji":"🟠","cause":"Fungus: Exserohilum turcicum","symptoms":"Long tan-grey cigar-shaped lesions on leaves.","treatment":["Apply Propiconazole or Tebuconazole","Spray at tasseling stage","Remove infected debris","Avoid dense planting"],"prevention":["Plant resistant hybrids","Rotate crops","Till soil to bury debris","Avoid excessive nitrogen"]},
    "corn cercospora leaf spot": {"name":"Corn Cercospora Leaf Spot","crop":"Corn/Maize","emoji":"🟡","cause":"Fungus: Cercospora zeae-maydis","symptoms":"Rectangular grey to tan lesions between leaf veins.","treatment":["Apply Strobilurin or Triazole fungicide","Spray at early onset","Improve drainage","Remove infected residue"],"prevention":["Plant resistant varieties","Rotate crops","Reduce plant density","Avoid minimum tillage"]},
    "corn healthy": {"name":"Corn Healthy","crop":"Corn/Maize","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "rice blast": {"name":"Rice Blast Disease","crop":"Rice","emoji":"🔴","cause":"Fungus: Magnaporthe oryzae","symptoms":"Diamond-shaped lesions with grey centers and brown borders.","treatment":["Apply Tricyclazole or Isoprothiolane","Spray at booting stage","Drain fields periodically","Remove infected debris"],"prevention":["Plant blast-resistant varieties","Avoid excessive nitrogen","Maintain proper water management","Use certified seeds"]},
    "rice brown spot": {"name":"Rice Brown Spot","crop":"Rice","emoji":"🟠","cause":"Fungus: Cochliobolus miyabeanus","symptoms":"Oval brown spots with yellow halo on leaves.","treatment":["Apply Mancozeb or Iprodione","Spray at tillering stage","Improve soil fertility","Remove infected debris"],"prevention":["Use certified seeds","Maintain proper nutrition","Avoid water stress","Rotate crops"]},
    "rice bacterial blight": {"name":"Rice Bacterial Blight","crop":"Rice","emoji":"🔴","cause":"Bacterium: Xanthomonas oryzae","symptoms":"Water-soaked lesions on leaf margins turning yellow then white.","treatment":["Apply copper-based bactericide","Drain fields","Remove infected plants","Avoid excessive nitrogen"],"prevention":["Plant resistant varieties","Use certified seeds","Avoid flood irrigation","Maintain field hygiene"]},
    "rice sheath blight": {"name":"Rice Sheath Blight","crop":"Rice","emoji":"🟠","cause":"Fungus: Rhizoctonia solani","symptoms":"Oval lesions on leaf sheaths near water line with brown borders.","treatment":["Apply Propiconazole or Hexaconazole","Spray at early tillering","Drain field","Remove infected stubble"],"prevention":["Reduce plant density","Avoid excessive nitrogen","Use resistant varieties","Rotate crops"]},
    "rice tungro": {"name":"Rice Tungro Disease","crop":"Rice","emoji":"🔴","cause":"Virus transmitted by green leafhopper","symptoms":"Yellow-orange discoloration of leaves, stunted growth.","treatment":["Control leafhopper with insecticide","Remove infected plants","Use systemic insecticide"],"prevention":["Plant tungro-resistant varieties","Control leafhopper","Synchronize planting dates","Remove infected plants early"]},
    "rice healthy": {"name":"Rice Healthy","crop":"Rice","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "cassava mosaic": {"name":"Cassava Mosaic Disease","crop":"Cassava","emoji":"🔴","cause":"Virus transmitted by whitefly Bemisia tabaci","symptoms":"Mosaic pattern of yellow and green on leaves, distortion, stunted growth.","treatment":["Remove and destroy infected plants","Control whitefly with insecticide","Use mineral oil spray","No chemical cure"],"prevention":["Plant certified virus-free cuttings","Use mosaic-resistant varieties","Control whitefly with neem","Inspect planting material"]},
    "cassava brown streak": {"name":"Cassava Brown Streak Disease","crop":"Cassava","emoji":"🔴","cause":"Virus: Cassava Brown Streak Virus","symptoms":"Yellow patches on leaves, brown streaks on stems, brown patches in tubers.","treatment":["Remove all infected plants","Control whitefly vectors","No chemical treatment","Replace with resistant varieties"],"prevention":["Use CBSD-resistant varieties","Plant certified cuttings","Control whitefly","Avoid moving material from infected areas"]},
    "cassava bacterial blight": {"name":"Cassava Bacterial Blight","crop":"Cassava","emoji":"🟠","cause":"Bacterium: Xanthomonas axonopodis","symptoms":"Angular water-soaked leaf spots, wilting, stem cankers.","treatment":["Apply copper-based bactericide","Remove infected parts","Disinfect cutting tools","Destroy severely infected plants"],"prevention":["Use disease-free planting material","Disinfect tools","Plant resistant varieties","Avoid infected soil"]},
    "cassava healthy": {"name":"Cassava Healthy","crop":"Cassava","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "groundnut early leaf spot": {"name":"Groundnut Early Leaf Spot","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Cercospora arachidicola","symptoms":"Dark brown circular spots on upper leaf with yellow halo.","treatment":["Spray Chlorothalonil or Mancozeb","Apply every 14 days","Remove infected leaves","Avoid overhead irrigation"],"prevention":["Use certified seeds","Rotate crops","Remove crop debris","Plant resistant varieties"]},
    "groundnut late leaf spot": {"name":"Groundnut Late Leaf Spot","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Cercosporidium personatum","symptoms":"Dark brown to black spots on lower leaf surface.","treatment":["Apply Tebuconazole or Propiconazole","Spray every 14 days","Remove infected material","Improve air circulation"],"prevention":["Rotate crops","Use resistant varieties","Remove crop debris","Avoid dense planting"]},
    "groundnut rosette": {"name":"Groundnut Rosette Disease","crop":"Groundnut","emoji":"🔴","cause":"Virus transmitted by aphid Aphis craccivora","symptoms":"Stunted plants with small mottled leaves, chlorotic rosette pattern.","treatment":["Control aphid with insecticide","Remove infected plants","Apply mineral oil to reduce spread"],"prevention":["Plant early","Use resistant varieties","Control aphid with neem","Plant barrier crops"]},
    "groundnut rust": {"name":"Groundnut Rust","crop":"Groundnut","emoji":"🟠","cause":"Fungus: Puccinia arachidis","symptoms":"Orange-brown pustules on lower leaf surfaces, yellowing.","treatment":["Apply Mancozeb or Propiconazole","Spray every 14 days","Remove infected leaves"],"prevention":["Plant resistant varieties","Rotate crops","Remove crop debris","Monitor fields"]},
    "pepper bacterial spot": {"name":"Pepper Bacterial Spot","crop":"Pepper","emoji":"🟠","cause":"Bacterium: Xanthomonas campestris","symptoms":"Small water-soaked spots turning brown with yellow halo.","treatment":["Apply copper bactericide every 7 days","Remove infected parts","Avoid wet plants"],"prevention":["Use certified seeds","Avoid overhead irrigation","Rotate crops","Disinfect tools"]},
    "pepper healthy": {"name":"Pepper Healthy","crop":"Pepper","emoji":"🟢","cause":"No disease","symptoms":"Plant appears completely healthy.","treatment":["No treatment needed"],"prevention":["Maintain good soil health","Monitor regularly"]},
    "onion purple blotch": {"name":"Onion Purple Blotch","crop":"Onion","emoji":"🟠","cause":"Fungus: Alternaria porri","symptoms":"Small white lesions with purple centers enlarging to kill leaves.","treatment":["Apply Mancozeb or Iprodione","Spray every 7-10 days","Remove infected leaves","Improve air circulation"],"prevention":["Use certified disease-free sets","Avoid overhead irrigation","Rotate crops","Remove crop debris"]},
    "apple scab": {"name":"Apple Scab","crop":"Apple","emoji":"🟠","cause":"Fungus: Venturia inaequalis","symptoms":"Olive-green to brown scab-like lesions on leaves and fruits.","treatment":["Apply Myclobutanil or Captan","Spray from bud break","Remove infected leaves","Prune for air circulation"],"prevention":["Plant resistant varieties","Remove fallen leaves","Apply dormant sprays","Prune for good airflow"]},
    "apple black rot": {"name":"Apple Black Rot","crop":"Apple","emoji":"🔴","cause":"Fungus: Botryosphaeria obtusa","symptoms":"Brown circular lesions on leaves, black rotting of fruits.","treatment":["Apply Captan or Thiophanate-methyl","Remove infected fruits","Spray every 7-10 days","Prune cankers"],"prevention":["Remove mummified fruits","Prune infected branches","Maintain tree vigor","Apply dormant copper spray"]},
    "grape black rot": {"name":"Grape Black Rot","crop":"Grape","emoji":"🔴","cause":"Fungus: Guignardia bidwellii","symptoms":"Brown circular lesions on leaves, shriveled black berries.","treatment":["Apply Myclobutanil or Mancozeb","Spray from bud break","Remove infected berries","Repeat every 7-14 days"],"prevention":["Remove mummified berries","Prune for air circulation","Apply early season sprays","Remove infected material"]},
}

# ── CSS ──────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.main { background: linear-gradient(180deg, #e8f5e9 0%, #f1f8f1 100%); }
.hero { background: linear-gradient(135deg, #1a5276 0%, #1e8449 50%, #27ae60 100%); color: white; padding: 40px 30px; border-radius: 20px; text-align: center; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(26,82,118,0.3); }
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0; }
.stat-card { background: white; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #27ae60; }
.stat-num { font-size: 2rem; font-weight: 700; color: #1a5276; }
.stat-label { color: #666; font-size: 0.85rem; }
.crop-badge { background: linear-gradient(135deg, #e8f5e9, #c8e6c9); border: 1px solid #a5d6a7; border-radius: 20px; padding: 8px 15px; display: inline-block; margin: 4px; font-size: 0.85rem; color: #1b5e20; font-weight: 500; }
.disease-result { background: linear-gradient(135deg, #fdecea, #fce4e4); border-left: 6px solid #e74c3c; border-radius: 12px; padding: 20px 25px; margin: 15px 0; box-shadow: 0 4px 15px rgba(231,76,60,0.15); }
.disease-result h2 { color: #c0392b; margin: 0 0 5px 0; }
.healthy-result { background: linear-gradient(135deg, #eafaf1, #d5f5e3); border-left: 6px solid #27ae60; border-radius: 12px; padding: 20px 25px; margin: 15px 0; }
.healthy-result h2 { color: #1e8449; margin: 0 0 5px 0; }
.error-result { background: linear-gradient(135deg, #fdf2e9, #fde8d8); border-left: 6px solid #e67e22; border-radius: 12px; padding: 20px 25px; margin: 15px 0; }
.error-result h2 { color: #d35400; margin: 0 0 5px 0; }
.confidence-card { background: white; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin: 15px 0; }
.confidence-num { font-size: 3rem; font-weight: 700; color: #1a5276; line-height: 1; }
.info-section { background: white; border-radius: 15px; padding: 20px 25px; margin: 12px 0; box-shadow: 0 4px 15px rgba(0,0,0,0.06); border-left: 4px solid #3498db; }
.info-section h4 { color: #1a5276; margin: 0 0 12px 0; }
.treatment-item { background: #e8f5e9; border-radius: 8px; padding: 8px 12px; margin: 6px 0; color: #1b5e20; font-size: 0.9rem; }
.prevention-item { background: #e3f2fd; border-radius: 8px; padding: 8px 12px; margin: 6px 0; color: #1565c0; font-size: 0.9rem; }
.footer { background: linear-gradient(135deg, #1a5276, #1e8449); color: white; text-align: center; padding: 20px; border-radius: 15px; margin-top: 30px; font-size: 13px; }
.footer a { color: #a9dfbf; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 Plant Disease Detection System</h1>
    <p style="font-size:1rem;">Federal University of Technology Babura</p>
    <p style="font-size:0.85rem;opacity:0.8;">
    🤖 AI-Powered | 🆓 Free | 📱 Works on Any Device</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown('<div class="stat-card"><div class="stat-num">14+</div><div class="stat-label">🌾 Crops</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="stat-card"><div class="stat-num">47+</div><div class="stat-label">🦠 Diseases</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="stat-card"><div class="stat-num">95%</div><div class="stat-label">🎯 Accuracy</div></div>', unsafe_allow_html=True)
with c4: st.markdown('<div class="stat-card"><div class="stat-num">🆓</div><div class="stat-label">💰 Free</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

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

    with st.expander("📖 How to use"):
        st.markdown("""
        **Step 1:** Upload a clear close-up photo of a plant leaf\n
        **Step 2:** Click the Detect Disease button\n
        **Step 3:** Read the diagnosis and follow treatment advice\n
        💡 **Tip:** Use a well-lit close-up photo for best results!
        """)

    st.markdown("### ✅ Crops Supported by AI Detection:")
    st.markdown("""
    <div style="background:white;padding:15px;border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-bottom:10px;">
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
    use <b>🔍 Search</b> or <b>🌾 Browse</b> tab!
    </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload a plant leaf image (JPG or PNG)",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        detect_btn = st.button(
            "🔍 DETECT DISEASE NOW",
            use_container_width=True,
            type="primary"
        )

        if detect_btn:
            if model_loaded and validator_loaded:

                # ── STAGE 1: Leaf Validation ───────────
                with st.spinner("🔍 Stage 1: Checking if image is a plant leaf..."):
                    img = image.resize((224,224))
                    img_array = np.array(img, dtype=np.float32)/255.0
                    img_input = np.expand_dims(img_array, axis=0)

                    validator.set_tensor(val_input[0]['index'], img_input)
                    validator.invoke()
                    val_pred = validator.get_tensor(val_output[0]['index'])
                    val_classes = ['leaf', 'not_leaf']
                    val_result = val_classes[np.argmax(val_pred)]
                    val_confidence = np.max(val_pred) * 100

                if val_result == 'not_leaf' and val_confidence > 70:
                    # ── NOT A LEAF ERROR ───────────────
                    st.markdown("""
                    <div class="error-result">
                        <h2>❌ Error: This is NOT a Plant Leaf!</h2>
                        <p>Our Computer Vision system has detected that the
                        uploaded image is not a plant leaf. This system only
                        works with plant leaf images.</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.error("""
                    🚫 **Invalid Image Detected!**

                    You may have uploaded:
                    - 👤 A photo of a person or human body
                    - 🐾 A photo of an animal
                    - 🏠 A photo of a building or object
                    - 📄 A blank or unclear image

                    **Please upload a clear photo of a plant leaf and try again!**
                    """)

                    st.markdown("""
                    <div style="background:#fff3e0;border-radius:12px;
                    padding:15px;margin-top:10px;border-left:4px solid #ff9800;">
                        <h4 style="color:#e65100;margin:0 0 10px 0;">
                        💡 Tips for a good leaf photo:</h4>
                        <p>✅ Take a close-up photo of the leaf only</p>
                        <p>✅ Make sure leaf fills most of the photo</p>
                        <p>✅ Use good natural lighting</p>
                        <p>✅ Keep the photo clear and sharp</p>
                        <p>❌ Do NOT upload photos of people or objects</p>
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    # ── STAGE 2: Disease Detection ─────
                    with st.spinner("🤖 Stage 2: AI is detecting the disease..."):
                        interpreter.set_tensor(input_details[0]['index'], img_input)
                        interpreter.invoke()
                        prediction = interpreter.get_tensor(output_details[0]['index'])
                        result = ai_classes[np.argmax(prediction)]
                        confidence = np.max(prediction) * 100

                    st.markdown("---")
                    st.markdown("## 📊 Detection Results")

                    # Show pipeline success
                    st.success("✅ Stage 1: Plant leaf confirmed! ✅ Stage 2: Disease analysis complete!")

                    if confidence < 60:
                        st.markdown("""
                        <div class="error-result">
                            <h2>⚠️ Unclear Result — Low Confidence!</h2>
                            <p>The AI could not clearly identify the disease.
                            Please try again with a clearer, well-lit photo.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.warning(f"⚠️ Confidence too low: {confidence:.1f}% — Please upload a clearer photo!")

                    elif "Healthy" in result:
                        st.markdown(f"""
                        <div class="healthy-result">
                            <h2>🟢 {result}</h2>
                            <p>Great news! Your plant appears completely healthy!
                            Keep monitoring regularly to maintain its health.</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="confidence-card">
                            <p style="color:#666;margin:0;">🎯 AI Confidence Score</p>
                            <div class="confidence-num">{confidence:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                        info = diseases_db.get(result.lower(), None)
                        if info:
                            st.markdown('<div class="info-section"><h4>🛡️ Maintenance Tips</h4>', unsafe_allow_html=True)
                            for p in info["prevention"]:
                                st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    else:
                        st.markdown(f"""
                        <div class="disease-result">
                            <h2>⚠️ Disease Detected: {result}</h2>
                            <p>Disease found! Please follow treatment
                            recommendations below immediately.</p>
                        </div>
                        """, unsafe_allow_html=True)

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
                            st.markdown(f"""
                            <div class="info-section">
                                <h4>🔬 Disease Information</h4>
                                <p><b>Disease:</b> {info['name']}</p>
                                <p><b>Crop:</b> {info['crop']}</p>
                                <p><b>Cause:</b> {info['cause']}</p>
                                <p><b>Symptoms:</b> {info['symptoms']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            st.markdown('<div class="info-section" style="border-left-color:#27ae60;"><h4>💊 Treatment</h4>', unsafe_allow_html=True)
                            for t in info["treatment"]:
                                st.markdown(f'<div class="treatment-item">✅ {t}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                            st.markdown('<div class="info-section" style="border-left-color:#3498db;"><h4>🛡️ Prevention</h4>', unsafe_allow_html=True)
                            for p in info["prevention"]:
                                st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.success("✅ Analysis complete! Follow the treatment steps above.")
            else:
                st.error("❌ Models not loaded. Please check deployment.")

# ════════════════════════════════════════════════
# TAB 2
# ════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Search Disease Information")
    st.write("Type any crop or disease name to get complete information instantly!")

    st.markdown("""
    <div style="background:#e8f5e9;border-radius:10px;padding:12px 15px;margin-bottom:15px;">
    💡 <b>Try:</b> tomato, rice, cassava, blight, rust, maize, groundnut, pepper
    </div>
    """, unsafe_allow_html=True)

    query = st.text_input("🔎 Type here:", placeholder="e.g. tomato, rice blast, cassava...")

    if query:
        q = query.lower().strip()
        found = [info for key, info in diseases_db.items()
                 if q in key or q in info["name"].lower()
                 or q in info["crop"].lower()
                 or q in info["symptoms"].lower()
                 or q in info["cause"].lower()]

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
                    st.markdown("**💊 Treatment:**")
                    for t in info["treatment"]:
                        st.markdown(f'<div class="treatment-item">✅ {t}</div>', unsafe_allow_html=True)
                    st.markdown("**🛡️ Prevention:**")
                    for p in info["prevention"]:
                        st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)
        else:
            st.error(f"❌ No results found for **'{query}'**")
            st.write("Try: **tomato, potato, rice, cassava, maize, groundnut, pepper, onion, apple, grape**")

# ════════════════════════════════════════════════
# TAB 3
# ════════════════════════════════════════════════
with tab3:
    st.markdown("## 🌾 Browse Diseases by Crop")
    st.write("Select any crop to see all its diseases and treatments!")

    crops = sorted(set(info["crop"] for info in diseases_db.values()))
    crop_emojis = {
        "Apple":"🍎","Cassava":"🌿","Corn/Maize":"🌽",
        "Grape":"🍇","Groundnut":"🥜","Onion":"🧅",
        "Pepper":"🌶️","Potato":"🥔","Rice":"🌾","Tomato":"🍅"
    }

    selected = st.selectbox("🌾 Select a crop:", ["-- Select a crop --"] + crops)

    if selected != "-- Select a crop --":
        matches = [info for info in diseases_db.values() if info["crop"] == selected]
        emoji = crop_emojis.get(selected, "🌿")

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a5276,#27ae60);
        color:white;padding:15px 20px;border-radius:12px;margin:10px 0;">
            <h3 style="margin:0;">{emoji} {selected}</h3>
            <p style="margin:5px 0 0 0;opacity:0.9;">
            Found <b>{len(matches)} diseases</b> in database</p>
        </div>
        """, unsafe_allow_html=True)

        healthy = [m for m in matches if "Healthy" in m["name"]]
        diseases = [m for m in matches if "Healthy" not in m["name"]]

        if healthy:
            st.markdown("### ✅ Healthy State")
            for info in healthy:
                with st.expander(f"🟢 {info['name']}"):
                    st.success("Plant is in healthy state — no disease detected.")
                    for p in info["prevention"]:
                        st.markdown(f'<div class="prevention-item">🔹 {p}</div>', unsafe_allow_html=True)

        if diseases:
            st.markdown(f"### ⚠️ {len(diseases)} Disease(s)")
            for info in diseases:
                with st.expander(f"{info['emoji']} {info['name']}"):
                    st.markdown(f"""
                    <div class="info-section">
                        <h4>🔬 Info</h4>
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
    <a href="https://futb-plant-disease.streamlit.app" style="color:#a9dfbf;">
    🌐 futb-plant-disease.streamlit.app</a>
</div>
""", unsafe_allow_html=True)
