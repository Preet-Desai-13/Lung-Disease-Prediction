import streamlit as st
import pickle
import numpy as np
import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Lung Disease Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── GLOBAL CSS ────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── App background ── */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .sidebar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    color: #58a6ff !important;
    letter-spacing: 0.02em;
    padding: 1rem 0 0.5rem 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 1.2rem;
}

/* ── Cards ── */
.card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 24px 28px;
    margin-bottom: 20px;
}
.card-sm {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* ── Badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-high   { background: #3d1a1a; color: #f85149; border: 1px solid #f85149; }
.badge-mod    { background: #2e2409; color: #e3b341; border: 1px solid #e3b341; }
.badge-low    { background: #0d2a1a; color: #3fb950; border: 1px solid #3fb950; }

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}

/* ── Stat number ── */
.stat-number {
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: -0.02em;
}
.stat-high  { color: #f85149; }
.stat-mod   { color: #e3b341; }
.stat-low   { color: #3fb950; }

/* ── Progress bar override ── */
.risk-track {
    height: 6px;
    background: #21262d;
    border-radius: 3px;
    overflow: hidden;
    margin-top: 10px;
}
.risk-fill { height: 100%; border-radius: 3px; }

/* ── FAQ button override ── */
div[data-testid="stVerticalBlock"] button[kind="secondary"] {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 8px 14px !important;
    min-width: unset !important;
    width: 100% !important;
    transition: border-color 0.15s, background 0.15s !important;
}
div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}

/* ── Primary action button ── */
.stButton > button[kind="primary"],
.stButton > button {
    background: #238636 !important;
    border: 1px solid #2ea043 !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    min-width: unset !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: none !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stSlider > div { color: #e6edf3 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-size: 0.85rem;
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #e6edf3 !important;
    border-bottom: 2px solid #58a6ff !important;
    background: transparent !important;
}

/* ── Divider ── */
hr { border-color: #30363d; }
.stDivider { display: none !important; }

/* ── Chat answer box ── */
.answer-box {
    background: #0d2233;
    border: 1px solid #1f4e79;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 16px 20px;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin-top: 12px;
}
.q-label {
    font-size: 0.75rem;
    color: #8b949e;
    margin-bottom: 6px;
}

/* ── Do / Dont rows ── */
.do-row   { color: #3fb950; font-size: 0.9rem; margin: 3px 0; }
.dont-row { color: #f85149; font-size: 0.9rem; margin: 3px 0; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    background: #161b22 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── FAQ DATA ──────────────────────────────
faqs = {
    "What are early signs of lung disease?": "Common early symptoms include persistent cough, shortness of breath, chest pain, and frequent lung infections.",
    "Is lung cancer risk higher for non-smokers?": "Genetic factors, air pollution, and secondhand smoke still pose risks to non-smokers.",
    "How does air pollution affect my lungs?": "Fine particulate matter can cause inflammation and long-term damage to respiratory tissues.",
    "Can a healthy diet improve lung health?": "Yes, antioxidants from fruits and vegetables help support cellular repair in the lungs.",
    "What is the role of genetics in lung risk?": "A family history of respiratory disease can double your overall risk profile.",
    "Does alcohol use impact lung health?": "Heavy alcohol consumption can weaken immunity, making lungs more susceptible to infections.",
    "Are occupational hazards dangerous?": "Long-term exposure to silica and chemical fumes is a leading cause of chronic pulmonary disease.",
    "What is the 'Confidence Score'?": "It represents the model's weighted certainty after analyzing all 25 clinical markers entered.",
    "Can children be at risk for lung disease?": "Yes, children in highly polluted areas face serious long-term respiratory development risks.",
    "How often should I get a lung screening?": "High-risk individuals should consult a doctor for screening every 12 months.",
    "What is shortness of breath called?": "In medical terms, shortness of breath is called 'Dyspnea'.",
    "Can pneumonia lead to long-term damage?": "Severe pneumonia can leave scarring in lung tissue known as pulmonary fibrosis.",
    "Does exercise help weak lungs?": "Cardiovascular exercise improves oxygen absorption efficiency in the lungs.",
    "How does passive smoking affect me?": "Secondhand smoke contains 7,000+ chemicals that damage lung tissue similar to active smoking.",
    "Is chest pain always related to lungs?": "Not always, but persistent discomfort during breathing is a strong pulmonary indicator.",
    "What is Chronic Bronchitis?": "Long-term inflammation of the bronchial tubes, often caused by smoking or air pollution.",
    "Can lung function be restored?": "Some damage is irreversible, but treatment can slow decline and improve breathing quality.",
    "Are vapes safer than cigarettes?": "Vapes still contain harmful chemicals that damage lung tissue and cause inflammation.",
    "What should I do if my risk is HIGH?": "Consult a Pulmonologist immediately and schedule a diagnostic CT scan as soon as possible.",
    "What is a wet vs dry cough?": "A dry cough is due to irritants; a wet cough usually indicates infection or fluid in the lungs.",
    "Can obesity increase lung disease risk?": "Yes, obesity reduces lung capacity and puts extra pressure on the respiratory system."
}

# ─────────────────────────── SESSION STATE ─────────────────────────
defaults = {
    "page": "Home",
    "patient_name": "",
    "faq_answer": None,
    "faq_question": None,
    "input_data": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.page = page
    st.rerun()

# ─────────────────────────── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🫁 Lung Disease Prediction</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.78rem; color:#8b949e; margin-bottom:1.5rem;'>Clinical diagnostic tool powered by Random Forest classification.</p>", unsafe_allow_html=True)

    nav_items = [("🏠  Home", "Home"), ("🩺  Run Diagnosis", "Dashboard"), ("📋  View Result", "Result")]
    for label, target in nav_items:
        active = st.session_state.page == target
        style = "color:#58a6ff !important; font-weight:600;" if active else ""
        if st.button(label, key=f"nav_{target}", use_container_width=True):
            go(target)

    st.markdown("<div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #30363d;'>", unsafe_allow_html=True)
    st.markdown("<p class='section-label'>About</p>", unsafe_allow_html=True)
    st.markdown("""
        <p style='font-size:0.78rem; color:#8b949e; line-height:1.6;'>
        This tool uses a trained ML model with 25 clinical features to assess lung disease risk.
        Always consult a certified medical professional for clinical decisions.
        </p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    # ── Hero ──
    col_h1, col_h2 = st.columns([1.6, 1], gap="large")
    with col_h1:
        st.markdown("<p class='section-label'>Clinical Decision Support</p>", unsafe_allow_html=True)
        st.markdown("""
            <h1 style='font-size:2.6rem; font-weight:700; line-height:1.2; color:#e6edf3; margin:0.2rem 0 1rem 0;'>
                Lung Disease<br>Risk Prediction
            </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
            <p style='font-size:1rem; color:#8b949e; line-height:1.7; max-width:480px;'>
                Enter patient data across three categories — profile, environmental exposure,
                and clinical symptoms — to receive an instant risk classification with actionable guidance.
            </p>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Run Diagnosis →"):
            go("Dashboard")

    with col_h2:
        st.markdown("""
            <div class='card' style='text-align:center; padding: 32px;'>
                <img src='https://img.icons8.com/clouds/200/lungs.png' style='width:180px; opacity:0.9;'>
                <p style='margin-top:16px; font-size:0.85rem; color:#8b949e;'>
                    Powered by <strong style='color:#58a6ff;'>Random Forest</strong> classifier<br>
                    trained on 25 clinical features
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── FAQ / Chatbot Section ──
    st.markdown("""
        <h2 style='font-size:1.3rem; font-weight:600; color:#e6edf3; margin-bottom:4px;'>
            Clinical Q&amp;A
        </h2>
        <p style='font-size:0.85rem; color:#8b949e; margin-bottom:1.5rem;'>
            Click any question to read the clinical answer.
        </p>
    """, unsafe_allow_html=True)

    faq_col1, faq_col2 = st.columns([1, 1.3], gap="large")
    with faq_col1:
        keys = list(faqs.keys())
        st.markdown("<p class='section-label'>Quick questions</p>", unsafe_allow_html=True)
        for q in keys[:5]:
            if st.button(q, key=f"faq_{q}", use_container_width=True):
                st.session_state.faq_question = q
                st.session_state.faq_answer = faqs[q]

        with st.expander("Show more questions"):
            for q in keys[5:]:
                if st.button(q, key=f"faq_{q}", use_container_width=True):
                    st.session_state.faq_question = q
                    st.session_state.faq_answer = faqs[q]

    with faq_col2:
        st.markdown("<p class='section-label'>Answer</p>", unsafe_allow_html=True)
        if st.session_state.faq_answer:
            st.markdown(f"""
                <p class='q-label'>→ {st.session_state.faq_question}</p>
                <div class='answer-box'>{st.session_state.faq_answer}</div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='border:1px dashed #30363d; border-radius:8px; padding:32px;
                            text-align:center; color:#484f58; font-size:0.88rem;'>
                    No question selected yet.<br>Click one on the left.
                </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.page == "Dashboard":
    st.markdown("""
        <h1 style='font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px;'>Run Diagnosis</h1>
        <p style='font-size:0.85rem; color:#8b949e; margin-bottom:1.5rem;'>
            Fill in the three sections below and submit to get a risk prediction.
        </p>
    """, unsafe_allow_html=True)

    try:
        model = pickle.load(open("model/lung_model.pkl", "rb"))
        feature_names = pickle.load(open("model/features.pkl", "rb"))
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        st.stop()

    in_v = {}
    f_p  = ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity"]
    f_h  = ["Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution"]
    f_signs = [f for f in feature_names if f not in f_p + f_h + ["index", "Patient Id"]]

    tab1, tab2, tab3 = st.tabs(["Patient Profile", "Environmental Exposure", "Symptoms"])

    with tab1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.session_state.patient_name = st.text_input("Patient full name", st.session_state.patient_name, placeholder="e.g. Ramesh Kumar")
        c1, c2, c3 = st.columns(3)
        with c1: in_v["Age"] = st.slider("Age", 0, 100, 30)
        with c2:
            gv = st.selectbox("Gender", ["Male", "Female", "Other"])
            in_v["Gender"] = 1 if gv == "Male" else 2
        with c3:
            in_v["Genetic Risk"] = 7 if st.selectbox("Family history of lung disease?", ["No", "Yes"]) == "Yes" else 2
        c4, c5 = st.columns(2)
        with c4: in_v["Balanced Diet"] = 7 if st.selectbox("Poor / unbalanced diet?", ["No", "Yes"]) == "Yes" else 2
        with c5: in_v["Obesity"]       = 7 if st.selectbox("Obesity?", ["No", "Yes"]) == "Yes" else 2
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_h):
            with [c1, c2, c3][i % 3]:
                in_v[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"env_{f}") == "Yes" else 2
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_signs):
            with [c1, c2, c3][i % 3]:
                in_v[f] = 7 if st.selectbox(f, ["No", "Yes"], key=f"sym_{f}") == "Yes" else 2
        st.markdown("</div>", unsafe_allow_html=True)

    for f in ["index", "Patient Id"]:
        if f in feature_names:
            in_v[f] = 0

    if st.button("Submit and Predict"):
        st.session_state.in_v = in_v
        st.session_state.input_data = np.array([[in_v[f] for f in feature_names]])
        go("Result")

# ═══════════════════════════════════════════════════════════════════
#  PAGE: RESULT
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.page == "Result":

    if st.session_state.input_data is None:
        st.warning("No prediction data found. Please run a diagnosis first.")
        if st.button("Go to Diagnosis"):
            go("Dashboard")
        st.stop()

    try:
        model = pickle.load(open("model/lung_model.pkl", "rb"))
        feature_names = pickle.load(open("model/features.pkl", "rb"))
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    probs  = model.predict_proba(st.session_state.input_data)[0]
    p_idx  = int(np.argmax(probs))
    perc   = probs[p_idx] * 100

    LABELS  = {0: "HIGH RISK",     1: "LOW RISK",     2: "MODERATE RISK"}
    COLORS  = {0: "#f85149",       1: "#3fb950",       2: "#e3b341"}
    BADGES  = {0: "badge-high",    1: "badge-low",     2: "badge-mod"}
    STAT_CL = {0: "stat-high",     1: "stat-low",      2: "stat-mod"}
    FILL_CL = {0: "background:#f85149;", 1: "background:#3fb950;", 2: "background:#e3b341;"}

    msg_map = {
        0: "High-severity clinical markers detected. An in-person consultation with a Pulmonologist is strongly advised without delay.",
        1: "No critical markers identified. Maintain healthy habits and schedule routine annual screenings.",
        2: "Moderate risk indicators present. A medical review is recommended within the next few weeks."
    }
    do_map = {
        0: ["Consult a Pulmonologist immediately", "Schedule a low-dose CT-Scan", "Monitor SpO₂ (oxygen) levels daily"],
        1: ["Maintain a balanced diet", "30 min cardio exercise, 5x/week", "Annual lung screening"],
        2: ["Book an appointment with a GP this week", "Track frequency of cough / chest discomfort", "Improve indoor air quality"]
    }
    dont_map = {
        0: ["Do not smoke or use tobacco in any form", "Avoid exposure to air pollutants", "Do not delay — seek care now"],
        1: ["Avoid extended exposure to secondhand smoke", "Do not skip future annual checkups"],
        2: ["Do not self-medicate", "Avoid highly polluted outdoor areas", "Do not ignore worsening symptoms"]
    }

    # ── Page header ──
    st.markdown(f"""
        <div style='display:flex; align-items:center; gap:14px; margin-bottom:1.5rem;'>
            <h1 style='font-size:1.6rem; font-weight:700; color:#e6edf3; margin:0;'>Prediction Result</h1>
            <span class='badge {BADGES[p_idx]}'>{LABELS[p_idx]}</span>
        </div>
        <p style='font-size:0.85rem; color:#8b949e; margin-bottom:1.5rem;'>
            Patient: <strong style='color:#c9d1d9;'>{st.session_state.patient_name or "N/A"}</strong>
            &nbsp;·&nbsp; {datetime.datetime.now().strftime("%d %b %Y, %H:%M")}
        </p>
    """, unsafe_allow_html=True)

    # ── Score card ──
    st.markdown(f"""
        <div class='card'>
            <p class='section-label'>Confidence Score</p>
            <div class='stat-number {STAT_CL[p_idx]}'>{perc:.1f}%</div>
            <p style='color:#8b949e; font-size:0.82rem; margin-top:6px;'>
                Model placed highest confidence in <strong>{LABELS[p_idx]}</strong> category
            </p>
            <div class='risk-track'>
                <div class='risk-fill' style='width:{perc:.1f}%; {FILL_CL[p_idx]}'></div>
            </div>
            <div style='display:flex; justify-content:space-between; margin-top:4px;'>
                <span style='font-size:0.7rem; color:#484f58;'>0%</span>
                <span style='font-size:0.7rem; color:#484f58;'>100%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Two columns: guidance + hospital ──
    r1, r2 = st.columns([1.2, 1], gap="large")

    with r1:
        st.markdown(f"""
            <div class='card'>
                <p class='section-label'>Clinical Summary</p>
                <p style='font-size:0.95rem; color:#c9d1d9; line-height:1.7; margin-bottom:20px;'>{msg_map[p_idx]}</p>
                <p class='section-label' style='margin-bottom:8px;'>Recommended Actions</p>
        """, unsafe_allow_html=True)
        for item in do_map[p_idx]:
            st.markdown(f"<p class='do-row'>✓ &nbsp;{item}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='section-label' style='margin-top:16px; margin-bottom:8px;'>Avoid / Restrictions</p>", unsafe_allow_html=True)
        for item in dont_map[p_idx]:
            st.markdown(f"<p class='dont-row'>✗ &nbsp;{item}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # PDF Download
        do_msg_str   = " | ".join(do_map[p_idx])
        dont_msg_str = " | ".join(dont_map[p_idx])

        def generate_pdf():
            buf = io.BytesIO()
            c   = canvas.Canvas(buf, pagesize=letter)
            c.setFillColor(colors.HexColor("#0d1117")); c.rect(0, 750, 612, 62, fill=1)
            c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 16)
            c.drawString(40, 788, "LUNG DISEASE PREDICTION — DIAGNOSTIC REPORT")
            c.setFont("Helvetica", 9); c.setFillColor(colors.HexColor("#8b949e"))
            c.drawString(40, 770, f"Generated: {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
            c.setFillColor(colors.HexColor("#30363d")); c.rect(40, 700, 532, 45, fill=1, stroke=0)
            c.setFillColor(colors.HexColor(COLORS[p_idx])); c.setFont("Helvetica-Bold", 14)
            c.drawString(52, 725, f"RESULT: {LABELS[p_idx]}")
            c.setFont("Helvetica", 11); c.setFillColor(colors.HexColor("#c9d1d9"))
            c.drawString(52, 710, f"Confidence: {perc:.1f}%")
            c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 11); c.drawString(40, 680, "Patient:")
            c.setFont("Helvetica", 11); c.drawString(110, 680, st.session_state.patient_name or "N/A")
            c.setFont("Helvetica-Bold", 11); c.drawString(40, 655, "Clinical Summary:")
            tb = c.beginText(40, 638); tb.setFont("Helvetica", 10); tb.textLines(msg_map[p_idx]); c.drawText(tb)
            c.setFont("Helvetica-Bold", 11); c.drawString(40, 610, "Recommended Actions:")
            tb2 = c.beginText(40, 594); tb2.setFont("Helvetica", 10); tb2.textLines(do_msg_str); c.drawText(tb2)
            c.setFont("Helvetica-Bold", 11); c.drawString(40, 570, "Restrictions:")
            tb3 = c.beginText(40, 554); tb3.setFont("Helvetica", 10); tb3.textLines(dont_msg_str); c.drawText(tb3)
            c.setFont("Helvetica-Oblique", 8); c.setFillColor(colors.HexColor("#8b949e"))
            c.drawString(40, 60, "This report is generated by a machine learning model for reference only. Consult a qualified physician.")
            c.save(); buf.seek(0); return buf

        pdf_buf = generate_pdf()
        st.download_button(
            label="Download PDF Report",
            data=pdf_buf,
            file_name=f"lung_report_{(st.session_state.patient_name or 'patient').replace(' ','_')}.pdf",
            mime="application/pdf"
        )

    with r2:
        st.markdown(f"""
            <div class='card'>
                <p class='section-label'>Find a Specialist</p>
                <p style='font-size:0.88rem; color:#8b949e; line-height:1.6; margin-bottom:20px;'>
                    Locate verified pulmonary clinics and respiratory specialists near you for a clinical evaluation.
                </p>
                <a href='https://www.google.com/maps/search/Lung+Specialist+Hospital+near+me' target='_blank'
                   style='display:block; background:#21262d; border:1px solid #30363d; border-radius:8px;
                          padding:14px 20px; color:#58a6ff; font-weight:600; font-size:0.9rem;
                          text-decoration:none; text-align:center;'>
                    📍 Find Nearest Hospital
                </a>
                <div style='margin-top:24px; padding-top:20px; border-top:1px solid #30363d;'>
                    <p style='font-size:0.78rem; color:#484f58; line-height:1.6;'>
                        <strong style='color:#8b949e;'>Disclaimer:</strong><br>
                        This tool is for informational purposes only and does not replace clinical diagnosis.
                        Always consult a certified medical professional before making health decisions.
                    </p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # All class probabilities
        st.markdown(f"""
            <div class='card'>
                <p class='section-label'>All Class Probabilities</p>
        """, unsafe_allow_html=True)
        class_labels = {0: "High Risk", 1: "Low Risk", 2: "Moderate Risk"}
        for idx, label in class_labels.items():
            prob_pct = probs[idx] * 100
            bar_color = COLORS[idx]
            st.markdown(f"""
                <div style='margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; font-size:0.8rem; color:#8b949e; margin-bottom:4px;'>
                        <span>{label}</span><span>{prob_pct:.1f}%</span>
                    </div>
                    <div class='risk-track'>
                        <div class='risk-fill' style='width:{prob_pct:.1f}%; background:{bar_color};'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
    <div style='margin-top:3rem; padding-top:1.5rem; border-top:1px solid #21262d;
                text-align:center; font-size:0.72rem; color:#484f58;'>
        Lung Disease Prediction &nbsp;·&nbsp; Clinical Suite v3.8 &nbsp;·&nbsp; For research use only
    </div>
""", unsafe_allow_html=True)