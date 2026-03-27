import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Lung Disease Prediction | AI Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME ENGINE (MATERIAL ELITE) --------------------
def apply_premium_ui():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Material+Icons&display=swap');
        
        :root {
            --primary: #00f2fe;
            --secondary: #4facfe;
            --danger: #ff4b2b;
            --glass-bg: rgba(25, 33, 50, 0.4);
            --glass-border: rgba(255, 255, 255, 0.08);
            --deep-gradient: linear-gradient(135deg, #020617 0%, #0f172a 50%, #020617 100%);
        }

        .stApp {
            background: var(--deep-gradient);
            background-attachment: fixed;
            color: #f8f9fa !important;
        }

        * { font-family: 'Poppins', sans-serif; }

        label, .stSlider p {
            font-size: 1.05rem !important;
            font-weight: 500 !important;
            color: rgba(255,255,255,0.8) !important;
        }

        .mi { font-family: 'Material Icons'; vertical-align: middle; font-size: 1.5rem; margin-right: 10px; }

        .glass-container {
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            margin-bottom: 30px;
        }
        
        .emergency-alert {
            background: rgba(255, 75, 43, 0.15);
            border: 2px solid var(--danger);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 25px;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 75, 43, 0.6); }
            70% { box-shadow: 0 0 0 20px rgba(255, 75, 43, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 75, 43, 0); }
        }

        .premium-nav {
            position: fixed; top: 0; left: 0; right: 0; height: 80px;
            background: rgba(2, 6, 23, 0.98);
            backdrop-filter: blur(15px);
            border-bottom: 1px solid var(--glass-border);
            z-index: 9999;
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 60px;
        }

        .section-header {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--secondary);
            margin: 20px 0 15px 0;
            display: flex; align-items: center;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            padding-bottom: 5px;
        }

        .stButton > button {
            background: linear-gradient(90deg, #00f2fe, #4facfe) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 12px 20px !important;
            font-weight: 700 !important;
            border: none !important;
            width: 100%;
            transition: 0.3s all ease-in-out;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(79, 172, 254, 0.4);
        }

        .logo-btn button {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #4facfe !important;
            width: auto !important;
        }
        
        /* Layout Fixes */
        .grid-group { margin-bottom: 20px; }
        hr, .stDivider { display: none !important; }
        #MainMenu, footer, header { visibility: hidden; }
        
        .result-perc {
            font-size: 3.5rem;
            font-weight: 900;
            margin: 0;
            background: linear-gradient(90deg, #fff, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)

apply_premium_ui()

# -------------------- KNOWLEDGE BASE (CHATBOT) --------------------
knowledge_base = {
    "What is lung cancer?": "Lung cancer is a disease where cells grow uncontrollably, forming tumors that impair breathing.",
    "Common symptoms?": "Persistent cough, shortness of breath, chest pain, and wheezing are critical signs.",
    "Smoking effects?": "Smoking causes chronic inflammation and accounts for over 80% of lung cancer cases.",
    "What is COPD?": "COPD blocks airflow and causes long-term breathing difficulty, usually triggered by smoking.",
    "Earliest signs?": "Fatigue, mild shortness of breath during activity, and a lingering dry cough.",
    "Is it reversible?": "Lung damage isn't fully reversible, but treatment can stop further progression.",
    "Genetic risks?": "Alpha-1 antitrypsin deficiency is a key genetic factor in early-onset lung issues."
}

# -------------------- GLOBAL STATE & NAVIGATION --------------------
if "page" not in st.session_state: st.session_state.page = "Home"
if "patient_name" not in st.session_state: st.session_state.patient_name = "Guest Patient"
if "history" not in st.session_state: st.session_state.history = []
if "chat_answer" not in st.session_state: st.session_state.chat_answer = None
if "view_more_chat" not in st.session_state: st.session_state.view_more_chat = False
if "show_info" not in st.session_state: st.session_state.show_info = None

def navigate_to(p):
    st.session_state.page = p
    st.rerun()

# --- SIDEBAR CHATBOT ---
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🏠 Lung Assistant</h2>", unsafe_allow_html=True)
    limit = len(knowledge_base) if st.session_state.view_more_chat else 4
    for i, (q, a) in enumerate(list(knowledge_base.items())[:limit]):
        if st.button(q, key=f"q_{i}"): st.session_state.chat_answer = a
    if not st.session_state.view_more_chat:
        if st.button("More FAQs +", key="v_m"): st.session_state.view_more_chat = True; st.rerun()
    if st.session_state.chat_answer:
        st.markdown(f"<div style='background:rgba(79,172,254,0.1); border:1px solid rgba(79,172,254,0.3); border-radius:15px; padding:15px; margin-top:15px;'><strong>Bot:</strong><br>{st.session_state.chat_answer}</div>", unsafe_allow_html=True)

# --- ELITE NAVBAR ---
st.markdown('<div class="premium-nav">', unsafe_allow_html=True)
col_l, col_r = st.columns([3, 2])
with col_l:
    if st.button("Lung Disease Prediction", key="logo"): navigate_to("Home")
with col_r:
    nb1, nb2, nb3 = st.columns(3)
    with nb1: 
        if st.button("Terminal", key="n_dash"): navigate_to("Dashboard")
    with nb2: 
        if st.button("Hub", key="n_about"): navigate_to("Home")
    with nb3: 
        if st.button("History", key="n_history"): navigate_to("Result")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    try: return pickle.load(open("model/lung_model.pkl", "rb")), pickle.load(open("model/features.pkl", "rb"))
    except: return None, None
model, feature_names = load_assets()

# --- PAGE: HOME ---
if st.session_state.page == "Home":
    hl, hr = st.columns([1, 1])
    with hl:
        st.markdown("<h1 style='font-size:4.5rem; font-weight:900; line-height:1; margin-top:100px;'>AI Lung <br><span style='color:#4facfe'>Diagnostic Hub</span></h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.3rem; opacity:0.8; margin-bottom:40px;'>Professional-grade AI clinical terminal for pulmonary risk assessments. Standardized for international clinical benchmarks.</p>", unsafe_allow_html=True)
        if st.button("Launch Diagnostic Terminal", key="launch_btn"): navigate_to("Dashboard")
    with hr:
        st.markdown("<div style='text-align:right; margin-top:80px;'><img src='https://img.icons8.com/clouds/300/lung.png' style='width:420px;'></div>", unsafe_allow_html=True)

# --- PAGE: DASHBOARD (REDESIGNED GRID LAYOUT) ---
elif st.session_state.page == "Dashboard":
    st.markdown('<div class="back-btn" style="margin-top:20px;">', unsafe_allow_html=True)
    if st.button("← Navigation Back", key="back_home_d"): navigate_to("Home")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<div class='glass-container' style='margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown("<h2 style='font-weight:700; margin-bottom:30px;'><span class='mi' style='color:#4facfe; font-size:2.5rem;'>biotech</span> Clinical Input Matrix</h2>", unsafe_allow_html=True)
    
    in_v = {}
    f_p = ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity"]
    f_h = ["Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution"]
    f_signs = [f for f in feature_names if f not in f_p + f_h + ["index", "Patient Id"]]
    
    tabs = st.tabs(["🧬 Patient Profile", "🌍 Environment & Habits", "🩺 Symptom Matrix"])
    
    with tabs[0]:
        st.markdown("<div class='section-header'>Registry Data</div>", unsafe_allow_html=True)
        st.session_state.patient_name = st.text_input("Full Patient Name", st.session_state.patient_name)
        c1, c2, c3 = st.columns(3)
        with c1: in_v["Age"] = st.slider("Patient Age", 0, 100, 30)
        with c2: 
            gv = st.selectbox("Designated Gender", ["Male", "Female", "Other"])
            in_v["Gender"] = 1 if gv == "Male" else 2
        with c3: 
            if gv == "Female": st.selectbox("Pregnancy Status", ["No", "Yes"], key="preg")
        
        st.markdown("<div class='section-header'>History Indicators</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: in_v["Genetic Risk"] = 7 if st.selectbox("Genetic Hist?", ["No", "Yes"]) == "Yes" else 2
        with c2: in_v["Balanced Diet"] = 7 if st.selectbox("Balanced Diet?", ["No", "Yes"]) == "Yes" else 2
        with c3: in_v["Obesity"] = 7 if st.selectbox("Obesity Hist?", ["No", "Yes"]) == "Yes" else 2

    with tabs[1]:
        st.markdown("<div class='section-header'>Exposure & Lifestyle Factors</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_h):
            with [c1, c2, c3][i % 3]: in_v[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"t2_{f}") == "Yes" else 2
            
    with tabs[2]:
        st.markdown("<div class='section-header'>Symptom Checklist (Grid View)</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_signs):
            with [c1, c2, c3][i % 3]: in_v[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"t3_{f}") == "Yes" else 2

    for f in ["index", "Patient Id"]: 
        if f in feature_names: in_v[f] = 0
        
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("EXECUTE AI DIAGNOSTIC SEQUENCE", use_container_width=True):
        st.session_state.in_v = in_v
        st.session_state.input_data = np.array([[in_v[f] for f in feature_names]])
        navigate_to("Result")
    st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE: RESULT (REDESIGNED PERCENTAGE FOCUS) ---
elif st.session_state.page == "Result":
    if st.button("← Terminal Home", key="back_home_r"): navigate_to("Home")
    
    if "input_data" in st.session_state and model:
        probs = model.predict_proba(st.session_state.input_data)[0]
        p_idx = np.argmax(probs)
        r_l = {0:"HIGH RISK", 1:"LOW RISK", 2:"MODERATE RISK"}
        r_c = {0:"#ff4b2b", 1:"#00f2fe", 2:"#f9d423"}
        perc = probs[p_idx] * 100
        
        if p_idx == 0:
            st.markdown(f"<div class='emergency-alert'><h2>🚨 EMERGENCY: {perc:.1f}% HIGH RISK</h2><p>Immediate medical attention required for {st.session_state.patient_name}.</p></div>", unsafe_allow_html=True)
        
        st.markdown(f"<div class='glass-container' style='border-left:12px solid {r_c[p_idx]}; text-align:center;'>", unsafe_allow_html=True)
        st.markdown(f"<p style='opacity:0.6; font-size:1.2rem; margin-bottom:10px;'>Diagnostic Conclusion for {st.session_state.patient_name}</p>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-perc' style='color:{r_c[p_idx]} !important;'>{perc:.1f}% {r_l[p_idx]}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='font-weight:700;'>🌍 Diagnostic Analytics</h2>", unsafe_allow_html=True)
        c1, c2 = st.columns([1.5, 1])
        with c1:
            st.markdown("<div class='glass-container' style='height:450px;'>", unsafe_allow_html=True)
            # Massive Gauge
            fig = go.Figure(go.Indicator(mode="gauge+number", value=perc, title={'text': "Comprehensive Risk Level"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': r_c[p_idx]}, 'steps': [{'range': [0, 30], 'color': "rgba(0, 242, 254, 0.1)"}, {'range': [30, 70], 'color': "rgba(249, 212, 35, 0.1)"}, {'range': [70, 100], 'color': "rgba(255, 75, 43, 0.1)"}]}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white', 'size': 18}, height=380)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='glass-container' style='height:450px;'><h4>🏥 Regional Support Hub</h4>", unsafe_allow_html=True)
            for h, s, d in [("Apollo Oncology", "Cancer Spec.", "1.2km"), ("AIIMS Center", "Pulmonary Spec.", "3.8km"), ("Max Health", "Respiratory Specialist", "5.1km")]:
                st.markdown(f"<div style='border-bottom:1px solid rgba(255,255,255,0.05); padding:10px 0;'><strong>{h}</strong><br><span style='font-size:0.85rem; opacity:0.7;'>{s} | {d}</span></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # History Tracking
        st.markdown("<h2 style='font-weight:700;'>📅 Diagnostic History Tracking</h2>", unsafe_allow_html=True)
        st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
        st.session_state.history.append({'Date': datetime.datetime.now().strftime("%I:%M %p"), 'Patient': st.session_state.patient_name, 'Conclusion': r_l[p_idx], 'Conf.': f"{perc:.1f}%"})
        st.table(pd.DataFrame(st.session_state.history).tail(5))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; opacity:0.3; font-size:0.8rem; margin:60px 0 20px 0;'>© 2026 AI Oncology Diagnostic Suite | Lung Disease Prediction AI</div>", unsafe_allow_html=True)