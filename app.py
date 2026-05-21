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
    "tomato early blight": {"name":"Tomato Early Blight","crop":"Tomato","emoji":"🟠","cause":"Fungus: Alternaria solani","symptoms":"Brown spots with yellow rings on lower leaves.","treatment":["Remove infected leaves immediately","Spray Chlorothalonil fungicide every 7 days","Apply copper-based fungicide as alternative","Water at base only"],"prevention":["Rotate crops every season","Plant resistant varieties","Proper spacing for air circulation","Remove plant debris after harvest"]},
    "tomato late blight": {"name":"Tomato Late Blight","crop":"Tomato","emoji":"🔴","cause":"Water mold: Phytophthora infestans","symptoms":"Dark watery spots on leaves and stems. White mold underneath.","treatment":["Remove and destroy infected plants immediately","Spray Mancozeb or Metalaxyl every 5-7 days","Do not compost — burn infected plants","Increase spray in wet weather"],"prevention":["Use drip irrigation only","Plant certified disease-free seeds","Ensure good drainage","Monitor during rainy season"]},
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

st.markdown("""
<style>
    .main{background-color:#f0f7f0}
    .title{background:linear-gradient(135deg,#1a5276,#27ae60);color:white;padding:30px;border-radius:15px;text-align:center;margin-bottom:20px}
    .disease-box{background:#fdecea;border-left:6px solid #e74c3c;padding:15px;border-radius:8px;margin:10px 0}
    .healthy-box{background:#eafaf1;border-left:6px solid #27ae60;padding:15px;border-radius:8px;margin:10px 0}
    .info-card{background:white;padding:20px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.1);margin:15px 0}
    .footer{text-align:center;color:#888;font-size:13px;margin-top:30px;padding:15px;border-top:1px solid #ddd}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title">
    <h1>🌿 Plant Disease Detection System</h1>
    <p style="font-size:16px;margin:0;">Federal University of Technology Babura</p>
    <p style="font-size:14px;margin:5px 0 0 0;opacity:0.85;">AI-Powered Early Detection for Healthier Crops</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("🌾 Crops", "10+")
col2.metric("🦠 Diseases", "47+")
col3.metric("🤖 Accuracy", "95.11%")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📸 AI Photo Detection", "🔍 Search by Name", "🌾 Browse by Crop"])

with tab1:
    st.markdown("### 📸 Upload Leaf Photo for AI Detection")
    st.write("Upload a plant leaf photo and let the AI detect the disease instantly!")

st.markdown("""
<div class="info-card">
<h4>🌿 Crops Supported by AI Detection:</h4>
<p>🍎 Apple &nbsp;|&nbsp; 🫐 Blueberry &nbsp;|&nbsp; 🍒 Cherry &nbsp;|&nbsp; 
🌽 Corn/Maize &nbsp;|&nbsp; 🍇 Grape &nbsp;|&nbsp; 🍊 Orange &nbsp;|&nbsp; 
🍑 Peach &nbsp;|&nbsp; 🌶️ Pepper &nbsp;|&nbsp; 🥔 Potato &nbsp;|&nbsp; 
🍓 Raspberry &nbsp;|&nbsp; 🫘 Soybean &nbsp;|&nbsp; 🎃 Squash &nbsp;|&nbsp; 
🍓 Strawberry &nbsp;|&nbsp; 🍅 Tomato</p>
<p><b>⚠️ Note:</b> For Rice, Cassava, Groundnut and Onion — use the 
<b>Search</b> or <b>Browse</b> tab instead!</p>
</div>
""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(image, caption="Uploaded Leaf", use_column_width=True)
        if st.button("🔍 Detect Disease", use_container_width=True):
            if model_loaded:
                with st.spinner("🤖 AI is analysing your leaf..."):
                    img = image.resize((224,224))
                    img = np.array(img, dtype=np.float32)/255.0
                    img = np.expand_dims(img, axis=0)
                    interpreter.set_tensor(input_details[0]['index'], img)
                    interpreter.invoke()
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    result = ai_classes[np.argmax(prediction)]
                    confidence = np.max(prediction)*100
                if "Healthy" in result:
                    st.markdown(f'<div class="healthy-box"><h2>🟢 {result}</h2><p>Your plant is healthy! Keep up the good work.</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="disease-box"><h2>⚠️ Disease Detected: {result}</h2><p>Please follow treatment recommendations below immediately.</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>📊 Confidence Score</h4><h2 style="color:#1a5276;text-align:center;">{confidence:.2f}%</h2></div>', unsafe_allow_html=True)
                info = diseases_db.get(result.lower(), None)
                if info:
                    st.markdown(f'<div class="info-card"><h4>🔬 Disease Information</h4><p><b>Cause:</b> {info["cause"]}</p><p><b>Symptoms:</b> {info["symptoms"]}</p></div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-card"><h4>💊 Treatment</h4>', unsafe_allow_html=True)
                    for t in info["treatment"]: st.write(f"✅ {t}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="info-card"><h4>🛡️ Prevention</h4>', unsafe_allow_html=True)
                    for p in info["prevention"]: st.write(f"🔹 {p}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ AI Model not loaded. Please check deployment.")

with tab2:
    st.markdown("### 🔍 Search Any Disease or Crop")
    st.write("Type any crop name or disease name to get full information!")
    query = st.text_input("Type here:", placeholder="e.g. tomato, rice blast, cassava, blight, rust, maize...")
    if query:
        q = query.lower().strip()
        found = [info for key, info in diseases_db.items() if q in key or q in info["name"].lower() or q in info["crop"].lower() or q in info["symptoms"].lower()]
        if found:
            st.success(f"✅ Found {len(found)} result(s) for '{query}'")
            for info in found:
                with st.expander(f"{info['emoji']} {info['name']} — {info['crop']}"):
                    st.write(f"**🔬 Cause:** {info['cause']}")
                    st.write(f"**👁️ Symptoms:** {info['symptoms']}")
                    st.write("**💊 Treatment:**")
                    for t in info["treatment"]: st.write(f"✅ {t}")
                    st.write("**🛡️ Prevention:**")
                    for p in info["prevention"]: st.write(f"🔹 {p}")
        else:
            st.warning(f"❌ No results found for '{query}'")
            st.write("Try: **tomato, potato, rice, cassava, maize, groundnut, pepper, onion, apple, grape**")

with tab3:
    st.markdown("### 🌾 Browse All Diseases by Crop")
    st.write("Select a crop to see all its diseases and treatments!")
    crops = sorted(set(info["crop"] for info in diseases_db.values()))
    selected = st.selectbox("Select a crop:", ["-- Select a crop --"] + crops)
    if selected != "-- Select a crop --":
        matches = [info for info in diseases_db.values() if info["crop"] == selected]
        st.info(f"Found **{len(matches)} diseases** for {selected}")
        for info in matches:
            with st.expander(f"{info['emoji']} {info['name']}"):
                st.write(f"**🔬 Cause:** {info['cause']}")
                st.write(f"**👁️ Symptoms:** {info['symptoms']}")
                st.write("**💊 Treatment:**")
                for t in info["treatment"]: st.write(f"✅ {t}")
                st.write("**🛡️ Prevention:**")
                for p in info["prevention"]: st.write(f"🔹 {p}")

st.markdown("""
<div class="footer">
    Developed by <b>Yusuf Gambo</b> | Matric: SIT/CSC/23/0005 | FUTB CS 2024/2025<br>
    Supervised by <b>Dr. Khalid Haruna</b> | 
    <a href="https://futb-plant-disease.streamlit.app" target="_blank">🌐 futb-plant-disease.streamlit.app</a>
</div>
""", unsafe_allow_html=True)
