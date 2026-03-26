import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="LungHealth AI | Global Oncology Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- PREMIUM GLASSMORPHISM CSS --------------------
def apply_premium_ui():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;600&display=swap');
        
        :root {
            --glass-bg: rgba(255, 255, 255, 0.08);
            --glass-border: rgba(255, 255, 255, 0.15);
            --glass-blur: blur(25px);
            --primary-gradient: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            --deep-gradient: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            --accent-glow: 0 0 20px rgba(79, 172, 254, 0.4);
        }

        .stApp {
            background: var(--deep-gradient);
            background-attachment: fixed;
            color: #ffffff !important;
        }

        * { font-family: 'Poppins', sans-serif; color: #f8f9fa; }

        .nav-container {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 80px;
            background: rgba(15, 12, 41, 0.9);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--glass-border);
            z-index: 9999;
            display: flex;
            align-items: center;
            padding: 0 40px;
        }

        div[data-testid="stHorizontalBlock"] .stButton > button {
            background: transparent !important;
            border: none !important;
            color: rgba(255, 255, 255, 0.7) !important;
            font-size: 1.1rem !important;
            text-transform: none !important;
            box-shadow: none !important;
        }
        div[data-testid="stHorizontalBlock"] .stButton > button:hover {
            color: #4facfe !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: var(--glass-blur);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            margin-bottom: 30px;
        }

        .stButton>button {
            background: var(--primary-gradient) !important;
            color: white !important;
            border-radius: 14px !important;
            font-weight: 600 !important;
            box-shadow: var(--accent-glow);
        }

        #MainMenu, footer, header {visibility: hidden;}
        .hero-title { font-size: 3.5rem; font-weight: 700; background: var(--primary-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        
        [data-testid="stSidebar"] {
            background: rgba(15, 12, 41, 0.95);
            border-right: 1px solid var(--glass-border);
        }
        </style>
    """, unsafe_allow_html=True)

apply_premium_ui()

# -------------------- NAVIGATION & STATE --------------------
if "page" not in st.session_state: st.session_state.page = "Home"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your LungHealth assistant. How can I help you today?"}]

def navigate_to(p):
    st.session_state.page = p
    st.rerun()

# -------------------- SIDEBAR CHAT --------------------
with st.sidebar:
    st.markdown("<h2 style='color:#4facfe;'>💬 Health Assistant</h2>", unsafe_allow_html=True)
    chat_box = st.container(height=350)
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
    
    kb = {
        "Early Symptoms?": "Cough, shortness of breath, chest pain, and weight loss are major early signs.",
        "Main Causes?": "Smoking, pollution, radon gas, and occupational hazards.",
        "How to prevent?": "Quit smoking, test for radon, and eat antioxidant-rich fruits.",
        "Treatment?": "Surgery, chemo, and targeted therapies based on stage."
    }
    
    st.markdown("### 💡 FAQ")
    for k, v in kb.items():
        if st.button(k, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": k})
            st.session_state.messages.append({"role": "assistant", "content": v})
            st.rerun()

# -------------------- NAVBAR --------------------
st.markdown('<div class="nav-container">', unsafe_allow_html=True)
n1, n2, n3, n4, n5 = st.columns([2.5, 1, 1, 1, 0.5])
with n1: st.markdown('<div style="font-weight: 700; font-size: 1.6rem; color: #4facfe; margin-top:10px;">🧬 LUNGHEALTH AI</div>', unsafe_allow_html=True)
with n2: 
    if st.button("Home", key="nh"): navigate_to("Home")
with n3: 
    if st.button("Analysis", key="na"): navigate_to("Analysis")
with n4: 
    if st.button("About Us", key="nab"): navigate_to("About")
st.markdown('</div><div style="margin-top: 100px;"></div>', unsafe_allow_html=True)

# -------------------- ASSETS --------------------
@st.cache_resource
def get_assets():
    try: return pickle.load(open("model/lung_model.pkl", "rb")), pickle.load(open("model/features.pkl", "rb"))
    except: return None, None
model, feature_names = get_assets()

# -------------------- ROUTING --------------------

if st.session_state.page == "Home":
    cl, cr = st.columns([1.3, 1])
    with cl:
        st.markdown("<h1 class='hero-title'>AI-Powered Lung Disease Detection</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.2rem; opacity:0.8;'>Precise oncology risk assessment using clinical-grade neural networks.</p>", unsafe_allow_html=True)
        if st.button("START SCAN"): navigate_to("Analysis")
    with cr:
        st.markdown("<div class='glass-card' style='text-align:center;'><img src='https://img.icons8.com/clouds/200/lung.png' style='width:200px;'><h3>Neural Precise</h3></div>", unsafe_allow_html=True)

elif st.session_state.page == "Analysis":
    st.markdown("<h2>🧪 Clinical Terminal</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    input_values = {}
    f_p = ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity"]
    f_h = ["Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution"]
    f_s = [f for f in feature_names if f not in f_p + f_h + ["index", "Patient Id"]]
    
    t1, t2, t3 = st.tabs(["🧬 PROFILE", "🏢 HABITS", "🤒 SYMPTOMS"])
    with t1:
        c1, c2 = st.columns(2)
        for i, f in enumerate(f_p):
            if f not in feature_names: continue
            with c1 if i%2==0 else c2:
                if f=="Age": input_values[f] = st.slider("Age", 0, 100, 30)
                elif f=="Gender": 
                    gv = st.selectbox("Gender", ["Male", "Female", "Other"])
                    input_values[f] = 1 if gv=="Male" else 2
                    if gv=="Female":
                        st.selectbox("Are you Pregnant?", ["No", "Yes"], key="preg_check")
                else: input_values[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"t1_{f}")=="Yes" else 2
    with t2:
        c1, c2 = st.columns(2)
        for i, f in enumerate(f_h):
            if f in feature_names:
                with c1 if i%2==0 else c2: input_values[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"t2_{f}")=="Yes" else 2
    with t3:
        c1, c2 = st.columns(2)
        for i, f in enumerate(f_s):
            if f in feature_names:
                with c1 if i%2==0 else c2: input_values[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"t3_{f}")=="Yes" else 2
    for f in ["index", "Patient Id"]:
        if f in feature_names: input_values[f] = 0
    st.markdown("</div>", unsafe_allow_html=True)
    if st.button("GENERATE AI DIAGNOSIS"):
        st.session_state.inputs = input_values
        st.session_state.input_data = np.array([[input_values[f] for f in feature_names]])
        navigate_to("Result")

elif st.session_state.page == "Result":
    if "input_data" in st.session_state and model:
        probs = model.predict_proba(st.session_state.input_data)[0]
        p_high, p_low, p_med = probs[0]*100, probs[1]*100, probs[2]*100
        pred_idx = np.argmax(probs)
        r_labels = {0: "High Risk", 1: "Low Risk", 2: "Medium Risk"}
        r_colors = {0: "#ff4b2b", 1: "#00f2fe", 2: "#f9d423"}
        
        st.markdown(f"<div class='glass-card' style='border-left:15px solid {r_colors[pred_idx]};'><h1>{r_labels[pred_idx]}</h1><p>Health Score: {100-p_low:.1f}%</p></div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<div class='glass-card'><h3>Risk Meter</h3>", unsafe_allow_html=True)
            fig = go.Figure(go.Indicator(mode="gauge+number", value=100-p_low, gauge={'axis':{'range':[0,100]}, 'bar':{'color':r_colors[pred_idx]}}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color':'white'}, height=280)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-card'><h3>Probabilities</h3>", unsafe_allow_html=True)
            st.write(f"🛑 High: {p_high:.1f}% | ⚠️ Med: {p_med:.1f}% | ✅ Low: {p_low:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)

        # RESTORING CHARTS
        st.divider()
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("<div class='glass-card'><h3>Symptom Severity</h3>", unsafe_allow_html=True)
            symp_list = [f for f in feature_names if f not in ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution", "index", "Patient Id"]]
            df_s = pd.DataFrame({"Symptom": symp_list[:7], "Intensity": [(st.session_state.inputs[f]/8)*100 for f in symp_list[:7]]})
            st.plotly_chart(px.bar(df_s, x='Intensity', y='Symptom', orientation='h', range_x=[0, 100], color='Intensity', color_continuous_scale='Reds'), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with sc2:
            st.markdown("<div class='glass-card'><h3>Top Risk Drivers</h3>", unsafe_allow_html=True)
            df_i = pd.DataFrame({"Factor": feature_names, "W": model.feature_importances_}).sort_values(by="W").tail(7)
            st.plotly_chart(px.bar(df_i, x="W", y="Factor", orientation='h', color_discrete_sequence=['#4facfe']), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # RESTORING DOWNLOAD
        st.divider()
        ac1, ac2 = st.columns(2)
        with ac1:
            st.markdown("#### 📥 Medical Report")
            p_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
            SimpleDocTemplate(p_path).build([Paragraph("LUNG CANER AI REPORT", getSampleStyleSheet()['Title']), Paragraph(f"Final Assessment: {r_labels[pred_idx]}", getSampleStyleSheet()['Normal'])])
            with open(p_path, "rb") as f: st.download_button("Download Assessment PDF", f, file_name="Report.pdf")
        with ac2:
            st.markdown("#### 🏥 Care Options")
            st.link_button("Find Specialists Near Me", "https://www.google.com/maps/search/Lung+Cancer+Hospital+near+me", use_container_width=True)

        if st.button("⬅️ NEW ANALYSIS"): navigate_to("Analysis")

elif st.session_state.page == "About":
    st.markdown("<div class='glass-card'><h1>About Us</h1><p>Our mission is to democratize high-end medical intelligence using Neural Vision.</p></div>", unsafe_allow_html=True)
    if st.button("HOME"): navigate_to("Home")

st.divider()
st.caption("© 2026 LungHealth AI Suite")