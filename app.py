import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Lung disease prediction | Clinical Suite",
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
            --warning: #f9d423;
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.1);
            --deep-gradient: radial-gradient(circle at top left, #0f172a 0%, #020617 100%);
        }

        .stApp {
            background: var(--deep-gradient);
            background-attachment: fixed;
            color: #f8f9fa !important;
        }

        * { font-family: 'Poppins', sans-serif; }

        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(25px);
            border-radius: 28px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            margin-bottom: 30px;
            border: 1px solid var(--glass-border);
            transition: all 0.3s ease;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, #00f2fe, #4facfe) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-weight: 700 !important;
            border: none !important;
            width: auto !important;
            min-width: 200px;
            font-size: 1.1rem !important;
        }
        .stButton > button:hover { transform: scale(1.02); }

        .hero-title {
            font-size: 5rem; font-weight: 800; line-height: 1; margin-bottom: 20px;
            letter-spacing: -2px; background: linear-gradient(to right, #fff, #4facfe, #00f2fe);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }

        .chat-response {
            background: rgba(0, 242, 254, 0.1);
            border-left: 5px solid var(--secondary);
            padding: 20px;
            border-radius: 0 15px 15px 0;
            margin-top: 20px;
            animation: fadeIn 0.5s ease;
        }

        .result-perc {
            font-size: 4rem; font-weight: 900; margin: 0;
            background: linear-gradient(90deg, #fff, #4facfe);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        
        .risk-bar { height: 12px; border-radius: 6px; background: linear-gradient(90deg, #00f2fe 0%, #f9d423 50%, #ff4b2b 100%); margin: 20px 0; position: relative; }
        .risk-marker { position: absolute; top: -6px; width: 24px; height: 24px; background: white; border: 4px solid var(--secondary); border-radius: 50%; transform: translateX(-50%); box-shadow: 0 0 15px rgba(255,255,255,0.5); }
        
        hr, .stDivider { display: none !important; }
        #MainMenu, footer, header { visibility: hidden; }
        </style>
    """, unsafe_allow_html=True)

apply_premium_ui()

# -------------------- FAQ KNOWLEDGE BASE --------------------
faqs = {
    "What are early signs of lung disease?": "Common early symptoms include persistent cough, shortness of breath, chest pain, and frequent lung infections like bronchitis.",
    "Is lung cancer risk higher for non-smokers?": "While smokers have the highest risk, non-smokers can still be affected by secondhand smoke, air pollution, and genetic factors.",
    "How does air pollution affect my lungs?": "Fine particulate matter (PM2.5) can enter deep into lung tissue, causing inflammation and long-term respiratory damage.",
    "Can a healthy diet improve lung health?": "Yes, antioxidants from fruits and vegetables help fight inflammation and support cellular repair in lung tissues.",
    "What is the role of genetics in lung risk?": "A family history of respiratory disease can double your risk, making regular diagnostic screenings essential.",
    "Does alcohol use impact lung health?": "Heavy alcohol consumption can weaken the immune system, making the lungs more susceptible to infections like pneumonia.",
    "Are occupational hazards dangerous?": "Yes, long-term exposure to asbestos, silica, and chemical fumes is a leading cause of chronic pulmonary disease.",
    "What is the 'Confidence Score' in this app?": "The Confidence Score represents the system's certainty after analyzing your specific clinical markers.",
    "Can children be at risk for lung disease?": "Yes, children in highly polluted areas or those exposed to secondhand smoke face serious long-term respiratory risks.",
    "How often should I get a lung screening?": "High-risk individuals should consult a doctor for a screening every 12 months, ideally with a Low-Dose CT scan.",
    "What is shortness of breath called?": "In medical terms, shortness of breath is called 'Dyspnea', and it is a major indicator of respiratory distress.",
    "Can pneumonia lead to long-term damage?": "Severe pneumonia can leave scarring in the lung tissue known as pulmonary fibrosis.",
    "What is the benefit of using this diagnostic terminal?": "Our system provides an instant, data-driven risk assessment that helps you decide when to seek professional help.",
    "Does exercise help weak lungs?": "Cardiovascular exercise improves the efficiency of oxygen absorption and strengthens respiratory muscles.",
    "How does passive smoking affect me?": "Secondhand smoke contains 7,000+ chemicals; long-term exposure carries nearly 30% of the risk of active smoking.",
    "Is chest pain always related to lungs?": "No, chest pain can be heart or muscle related, but persistent discomfort during breathing is a strong lung marker.",
    "What is Chronic Bronchitis?": "It is a long-term inflammation of the bronchial tubes, often caused by smoking or prolonged air pollution exposure.",
    "Can lung function be restored?": "Some damage is irreversible, but quitting smoking and medical treatment can stop further decline and improve breathing.",
    "Are vapes safer than cigarettes?": "While they lack tobacco smoke, vapes still contain harmful chemicals that damage lung tissue and cause inflammation.",
    "What should I do if my risk is HIGH?": "Consult a Pulmonologist immediately, avoid all pollutants, and schedule a diagnostic CT scan as soon as possible.",
    "Is a dry cough different from a wet cough?": "A dry cough is often due to irritants or allergies, while a wet cough usually indicates infection or fluid in the lungs."
}

# -------------------- GLOBAL STATE & NAVIGATION --------------------
if "page" not in st.session_state: st.session_state.page = "Home"
if "patient_name" not in st.session_state: st.session_state.patient_name = "Guest"
if "faq_answer" not in st.session_state: st.session_state.faq_answer = None
if "faq_question" not in st.session_state: st.session_state.faq_question = None

def navigate_to(p):
    st.session_state.page = p
    st.rerun()

# --- NAVBAR ---
st.markdown('<div style="height:100px; display:flex; align-items:center; padding:0 60px;">', unsafe_allow_html=True)
st.markdown('<h2 style="margin:0; background:linear-gradient(90deg, #fff, #4facfe); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Lung disease prediction</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------- PAGE: HOME (W/ CHATBOT) --------------------
if st.session_state.page == "Home":
    st.markdown("<div style='margin-bottom: 80px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.markdown("<div class='hero-title'>Lung disease prediction <br>Terminal</div>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.4rem; opacity:0.6; margin-bottom:40px;'>Intelligent clinical risk assessment hub. Faster, smarter, professional.</p>", unsafe_allow_html=True)
        if st.button("🚀 Enter Diagnostic Terminal"): navigate_to("Dashboard")
    with col2:
        st.markdown("<div style='text-align:right;'><img src='https://img.icons8.com/clouds/500/lungs.png' style='width:550px; filter: drop-shadow(0 0 50px rgba(79, 172, 254, 0.3));'></div>", unsafe_allow_html=True)

    st.markdown("<br><br><br><h2 style='text-align:center; font-weight:700; margin-bottom:50px;'>💬 Lung Health Intelligence Hub</h2>", unsafe_allow_html=True)
    q_col1, q_col2 = st.columns([0.8, 1.2])
    with q_col1:
        st.markdown("<p style='opacity:0.6; font-size:1.1rem; margin-bottom:20px;'>Quick Clinical Queries:</p>", unsafe_allow_html=True)
        keys = list(faqs.keys())
        for q in keys[:5]:
            if st.button(q, key=f"faq_{q}", use_container_width=True):
                st.session_state.faq_question = q
                st.session_state.faq_answer = faqs[q]
        with st.expander("➕ View More Questions"):
            for q in keys[5:]:
                if st.button(q, key=f"faq_{q}", use_container_width=True):
                    st.session_state.faq_question = q
                    st.session_state.faq_answer = faqs[q]
    with q_col2:
        if st.session_state.faq_answer:
            st.markdown(f"""
                <div class='glass-card' style='min-height:300px;'>
                    <h4 style='color:#4facfe;'>🤖 Intelligence Bot Response</h4>
                    <div class='chat-response'>
                        <p style='color:grey; font-size:0.8rem; margin-bottom:5px;'>Question: <strong>{st.session_state.faq_question}</strong></p>
                        <p style='font-size:1.1rem; line-height:1.6;'>{st.session_state.faq_answer}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='height:400px; display:flex; align-items:center; justify-content:center; opacity:0.3;'>Select a question on the left to activate medical intelligence.</div>", unsafe_allow_html=True)

# --- PAGE: DASHBOARD ---
elif st.session_state.page == "Dashboard":
    if st.button("← Back to Hub"): navigate_to("Home")
    st.markdown("<h2 style='font-weight:700; padding: 20px 0;'>⚙️ Diagnostic Configuration</h2>", unsafe_allow_html=True)
    try: model, feature_names = pickle.load(open("model/lung_model.pkl", "rb")), pickle.load(open("model/features.pkl", "rb"))
    except: st.error("Model Error."); st.stop()
    in_v = {}
    f_p, f_h = ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity"], ["Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution"]
    f_signs = [f for f in feature_names if f not in f_p + f_h + ["index", "Patient Id"]]
    t = st.tabs(["👤 Patient Info", "🌍 Exposure Profile", "🩺 Symptom Check"])
    with t[0]:
        st.session_state.patient_name = st.text_input("Full Patient Name", st.session_state.patient_name)
        c1, c2, c3 = st.columns(3)
        with c1: in_v["Age"] = st.slider("Step 1: Participant Age", 0, 100, 30); in_v["Gender"] = 1 if st.selectbox("Assign Gender", ["Male", "Female", "Other"]) == "Male" else 2
        with c2: in_v["Genetic Risk"] = 7 if st.selectbox("Family History?", ["No", "Yes"]) == "Yes" else 2; in_v["Balanced Diet"] = 7 if st.selectbox("Balanced diet deficiency?", ["No", "Yes"]) == "Yes" else 2
        with c3: in_v["Obesity"] = 7 if st.selectbox("Obesity clinical monitor?", ["No", "Yes"]) == "Yes" else 2
    for f in f_h + f_signs: in_v[f] = 7 if st.sidebar.checkbox(f, False, key=f"side_{f}") else 2 # Sidebar for faster input
    for f in ["index", "Patient Id"]: 
        if f in feature_names: in_v[f] = 0
    if st.button("RUN PREDICTION ANALYSIS →"):
        st.session_state.input_data = np.array([[in_v[f] for f in feature_names]])
        navigate_to("Result")

# --- PAGE: RESULT ---
elif st.session_state.page == "Result":
    if st.button("← New Assessment"): navigate_to("Dashboard")
    if "input_data" in st.session_state:
        try: model, feature_names = pickle.load(open("model/lung_model.pkl", "rb")), pickle.load(open("model/features.pkl", "rb"))
        except: st.error("Model Error."); st.stop()
        probs = model.predict_proba(st.session_state.input_data)[0]
        p_idx = np.argmax(probs)
        r_l, r_c = {0:"HIGH RISK", 1:"LOW RISK", 2:"MODERATE RISK"}, {0:"#ff4b2b", 1:"#00f2fe", 2:"#f9d423"}
        perc = probs[p_idx] * 100
        
        st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <p style='opacity:0.6; font-size:1.1rem;'>Clinical Summary for: <strong>{st.session_state.patient_name}</strong></p>
                <div class='result-perc' style='color:{r_c[p_idx]} !important;'>{perc:.1f}% {r_l[p_idx]}</div>
                <div style='max-width:600px; margin: 30px auto;'><div class='risk-bar'><div class='risk-marker' style='left: {perc if p_idx==0 else (perc/2 if p_idx==2 else perc/4)}%;'></div></div></div>
                <p style='color:{r_c[p_idx]}; font-size:1.1rem; font-weight:700;'>SYSTEM CONFIDENCE SCORE: {perc:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            msg = {0: "CRITICAL ALERT: Emergency specialist consultation required.", 1: "STABLE: Healthy diagnostic range detected.", 2: "CAUTION: Moderate risks identified. Medical review recommended."}
            do_msg = {0: "✅ Consult Pulmonologist | ✅ Get CT-Scan", 1: "✅ Cardio exercise | ✅ Yearly checkup", 2: "✅ Consult specialist | ✅ Track cough"}
            dont_msg = {0: "❌ NO smoking | ❌ NO pollution", 1: "❌ NO passive smoking", 2: "❌ NO self-medication"}

            st.markdown(f"<div class='glass-card' style='min-height:480px;'><h3>🩺 Diagnostic Conclusion</h3><div style='padding:20px; background:rgba(255,255,255,0.02); border-left:5px solid {r_c[p_idx]}; border-radius:15px; margin-bottom:25px;'><p style='font-size:1.2rem; font-weight:600; color:{r_c[p_idx]};'>{msg[p_idx]}</p></div><h5 style='color:#4facfe;'>✅ WHAT TO DO:</h5><p>{do_msg[p_idx]}</p><h5 style='color:#ff4b2b;'>❌ WHAT NOT TO DO:</h5><p>{dont_msg[p_idx]}</p></div>", unsafe_allow_html=True)
            
            # --- ACTUAL PDF GENERATION ---
            def get_pdf_download():
                buf = io.BytesIO()
                ca = canvas.Canvas(buf, pagesize=letter)
                ca.setFillColor(colors.HexColor("#0f172a")); ca.rect(0, 750, 612, 50, fill=1)
                ca.setFillColor(colors.white); ca.setFont("Helvetica-Bold", 18); ca.drawString(40, 770, "LUNG DISEASE REPORT")
                ca.setFillColor(colors.black); ca.line(40, 730, 572, 730)
                ca.setFont("Helvetica-Bold", 12); ca.drawString(40, 710, "PATIENT PROFILE")
                ca.setFont("Helvetica", 11); ca.drawString(40, 690, f"Name: {st.session_state.patient_name}"); ca.drawString(40, 675, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
                ca.setFillColor(colors.HexColor(r_c[p_idx])); ca.setFont("Helvetica-Bold", 16); ca.drawString(40, 630, f"CONCLUSION: {r_l[p_idx]} ({perc:.1f}%)")
                ca.setFillColor(colors.black); ca.setFont("Helvetica-Bold", 12); ca.drawString(40, 600, "ADVISORY")
                t = ca.beginText(40, 580); t.setFont("Helvetica", 10); t.textLine(f"Do: {do_msg[p_idx]}"); t.textLine(f"Dont: {dont_msg[p_idx]}"); ca.drawText(t)
                ca.save()
                buf.seek(0)
                return buf.getvalue()

            st.download_button(label="📄 Download Official PDF Report", data=get_pdf_download(), file_name=f"Lung_Report_{st.session_state.patient_name}.pdf", mime="application/pdf")

        with c2:
            st.markdown(f"<div class='glass-card' style='min-height:510px;'><h3>📍 Healthcare Access</h3><p style='opacity:0.8;'>Connect with verified specialists near your location.</p><a href='https://www.google.com/maps/search/Lung+Specialist+Hospital+near+me' style='text-decoration:none;'><div style='background:linear-gradient(90deg, #00f2fe, #4facfe); padding:20px; border-radius:12px; color:white; font-weight:800; font-size:1.2rem; text-align:center;'>📍 FIND NEAREST HOSPITAL</div></a></div>", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; opacity:0.1; font-size:0.7rem; padding:60px;'>LUNG DISEASE PREDICTION v3.7.5 | Clinical Suite</div>", unsafe_allow_html=True)