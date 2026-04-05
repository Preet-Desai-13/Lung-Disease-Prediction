import streamlit as st
import pickle
import numpy as np
import pandas as pd
import datetime
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Lung Disease Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
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
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }
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

/* ── Step indicator ── */
@keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(46,160,67,0.55); }
    70%  { box-shadow: 0 0 0 10px rgba(46,160,67,0); }
    100% { box-shadow: 0 0 0 0 rgba(46,160,67,0); }
}
.step-bar-wrapper {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
}
.step-bar {
    display: flex;
    align-items: center;
    gap: 0;
    width: 100%;
}
.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    position: relative;
}
.step-circle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 700;
    flex-shrink: 0;
    transition: all 0.3s ease;
}
.step-active {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: 2px solid #3fb950;
    box-shadow: 0 0 0 4px rgba(46,160,67,0.2);
    animation: pulse-ring 1.8s ease-out infinite;
}
.step-done {
    background: linear-gradient(135deg, #1a7f3c, #238636);
    color: #d2ffd9;
    border: 2px solid #3fb950;
    box-shadow: 0 0 12px rgba(63,185,80,0.3);
}
.step-pending {
    background: #21262d;
    color: #484f58;
    border: 2px solid #30363d;
}
.step-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    white-space: nowrap;
    text-align: center;
}
.step-label-active  { color: #e6edf3; }
.step-label-done    { color: #3fb950; }
.step-label-pending { color: #484f58; }
.step-connector {
    flex: 1;
    height: 3px;
    border-radius: 2px;
    margin: 0 10px;
    margin-bottom: 20px;
    background: #21262d;
    position: relative;
    overflow: hidden;
}
.step-connector-done {
    background: linear-gradient(90deg, #238636, #3fb950);
    box-shadow: 0 0 6px rgba(63,185,80,0.4);
}

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

/* ── Back/Home button style ── */
.nav-btn-row {
    display: flex;
    gap: 10px;
    margin-bottom: 1.2rem;
    align-items: center;
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

/* ── Download button special style ── */
.stDownloadButton > button {
    background: #1f4e79 !important;
    border: 1px solid #1f6feb !important;
    color: #58a6ff !important;
}
.stDownloadButton > button:hover {
    background: #1f6feb !important;
    color: white !important;
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
    "form_step": 1,          # 1 = Patient Profile, 2 = Env Exposure, 3 = Symptoms
    "in_v": {},
    "pdf_ready": False,
    "pdf_bytes": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.page = page
    st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ═══════════════════════════════════════════════════════════════════
if st.session_state.page == "Home":

    # ── Hero ──
    col_h1, col_h2 = st.columns([1.6, 1], gap="large")
    with col_h1:
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
        if st.button("Lung Disease Prediction →", key="hero_start"):
            st.session_state.form_step = 1
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

        # Toggle button to open/close extra questions
        if "faq_expander_open" not in st.session_state:
            st.session_state.faq_expander_open = False
        toggle_label = "🔼 Hide extra questions" if st.session_state.faq_expander_open else "🔽 Show more questions"
        if st.button(toggle_label, key="faq_toggle", use_container_width=True):
            st.session_state.faq_expander_open = not st.session_state.faq_expander_open
            st.rerun()

        if st.session_state.faq_expander_open:
            for q in keys[5:]:
                if st.button(q, key=f"faq_{q}", use_container_width=True):
                    st.session_state.faq_question = q
                    st.session_state.faq_answer = faqs[q]
                    st.session_state.faq_expander_open = True

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
#  PAGE: DASHBOARD  (3-step wizard with Next / Back / Predict)
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.page == "Dashboard":

    # ── Top nav row ──
    nav_col1, nav_col2, nav_spacer = st.columns([1, 1, 6])
    with nav_col1:
        if st.button("Home", key="dash_home"):
            go("Home")
    with nav_col2:
        if st.session_state.input_data is not None:
            if st.button("View Result", key="dash_result"):
                go("Result")

    # ── Title ──
    st.markdown("""
        <h1 style='font-size:1.6rem; font-weight:700; color:#e6edf3; margin-bottom:4px;'>Run Diagnosis</h1>
        <p style='font-size:0.85rem; color:#8b949e; margin-bottom:1.5rem;'>
            Complete all three steps below and click Predict to get your risk assessment.
        </p>
    """, unsafe_allow_html=True)

    try:
        model = pickle.load(open("model/lung_model.pkl", "rb"))
        feature_names = pickle.load(open("model/features.pkl", "rb"))
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        st.stop()

    f_p    = ["Age", "Gender", "Genetic Risk", "Balanced Diet", "Obesity"]
    f_h    = ["Smoking", "Passive Smoker", "Alcohol use", "OccuPational Hazards", "Air Pollution"]
    f_signs = [f for f in feature_names if f not in f_p + f_h + ["index", "Patient Id"]]

    step = st.session_state.form_step

    # ── Step indicator ──
    step_names = ["Patient Profile", "Environmental Exposure", "Symptoms"]

    def step_class(i):
        if i < step:   return "step-done",    "step-label-done"
        if i == step:  return "step-active",   "step-label-active"
        return "step-pending", "step-label-pending"

    step_html = "<div class='step-bar-wrapper'><div class='step-bar'>"
    for i in range(1, 4):
        circ_cls, lbl_cls = step_class(i)
        icon = "✓" if i < step else str(i)
        step_html += f"""
            <div class='step-item'>
                <div class='step-circle {circ_cls}'>{icon}</div>
                <span class='step-label {lbl_cls}'>{step_names[i-1]}</span>
            </div>
        """
        if i < 3:
            conn_cls = "step-connector-done" if i < step else ""
            step_html += f"<div class='step-connector {conn_cls}'></div>"
    step_html += "</div></div>"
    st.markdown(step_html, unsafe_allow_html=True)

    # ── Restore previously entered values ──
    in_v = st.session_state.in_v if st.session_state.in_v else {}

    # ────────────────  STEP 1: Patient Profile  ────────────────
    if step == 1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-label' style='margin-bottom:14px;'>Step 1 — Patient Profile</p>", unsafe_allow_html=True)

        patient_name = st.text_input(
            "Patient full name *",
            value=st.session_state.patient_name,
            placeholder=""
        )
        st.session_state.patient_name = patient_name

        c1, c2, c3 = st.columns(3)
        with c1:
            age_str = st.text_input(
                "Age *",
                value=str(in_v["Age"]) if "Age" in in_v else "",
                placeholder="Enter age"
            )
            if age_str.strip():
                if age_str.strip().isdigit() and 1 <= int(age_str.strip()) <= 100:
                    in_v["Age"] = int(age_str.strip())
                else:
                    st.error("Age must be a number between 1 and 100.")
            elif "Age" in in_v:
                del in_v["Age"]
        with c2:
            gender_options = ["-- Select --", "Male", "Female", "Other"]
            if "Gender" in in_v:
                saved_gender = "Male" if in_v["Gender"] == 1 else ("Female" if in_v["Gender"] == 2 else "Other")
                g_idx = gender_options.index(saved_gender)
            else:
                g_idx = 0
            gv = st.selectbox("Gender *", gender_options, index=g_idx)
            in_v["Gender"] = 1 if gv == "Male" else (2 if gv == "Female" else (3 if gv == "Other" else 0))
        with c3:
            gr_opts = ["No", "Yes"]
            gr_saved = "Yes" if in_v.get("Genetic Risk", 2) == 7 else "No"
            in_v["Genetic Risk"] = 7 if st.selectbox("Family history of lung disease? *", gr_opts,
                                                       index=gr_opts.index(gr_saved)) == "Yes" else 2

        c4, c5 = st.columns(2)
        with c4:
            bd_opts = ["No", "Yes"]
            bd_saved = "Yes" if in_v.get("Balanced Diet", 2) == 7 else "No"
            in_v["Balanced Diet"] = 7 if st.selectbox("Poor / unbalanced diet? *", bd_opts,
                                                        index=bd_opts.index(bd_saved)) == "Yes" else 2
        with c5:
            ob_opts = ["No", "Yes"]
            ob_saved = "Yes" if in_v.get("Obesity", 2) == 7 else "No"
            in_v["Obesity"] = 7 if st.selectbox("Obesity? *", ob_opts,
                                                  index=ob_opts.index(ob_saved)) == "Yes" else 2

        st.markdown("</div>", unsafe_allow_html=True)

        _, btn_col = st.columns([6, 1])
        with btn_col:
            if st.button("Next →", key="step1_next"):
                if not patient_name.strip():
                    st.error("⚠️ Please enter the patient's full name before proceeding.")
                elif "Age" not in in_v or in_v.get("Age") is None:
                    st.error("⚠️ Please enter the patient's age before proceeding.")
                elif in_v.get("Gender", 0) == 0:
                    st.error("⚠️ Please select a Gender before proceeding.")
                else:
                    st.session_state.in_v = in_v
                    st.session_state.form_step = 2
                    st.rerun()

    # ────────────────  STEP 2: Environmental Exposure  ────────────────
    elif step == 2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-label' style='margin-bottom:14px;'>Step 2 — Environmental Exposure</p>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_h):
            with [c1, c2, c3][i % 3]:
                opts = ["No", "Yes"]
                saved = "Yes" if in_v.get(f, 2) == 7 else "No"
                in_v[f] = 7 if st.selectbox(f, opts, index=opts.index(saved), key=f"env_{f}") == "Yes" else 2

        st.markdown("</div>", unsafe_allow_html=True)

        back_col, _, next_col = st.columns([1, 5, 1])
        with back_col:
            if st.button("← Back", key="step2_back"):
                st.session_state.in_v = in_v
                st.session_state.form_step = 1
                st.rerun()
        with next_col:
            if st.button("Next →", key="step2_next"):
                st.session_state.in_v = in_v
                st.session_state.form_step = 3
                st.rerun()

    # ────────────────  STEP 3: Symptoms + Predict  ────────────────
    elif step == 3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='section-label' style='margin-bottom:14px;'>Step 3 — Clinical Symptoms</p>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        for i, f in enumerate(f_signs):
            with [c1, c2, c3][i % 3]:
                opts = ["No", "Yes"]
                saved = "Yes" if in_v.get(f, 2) == 7 else "No"
                in_v[f] = 7 if st.selectbox(f, opts, index=opts.index(saved), key=f"sym_{f}") == "Yes" else 2

        st.markdown("</div>", unsafe_allow_html=True)

        # Fill dummy fields
        for f in ["index", "Patient Id"]:
            if f in feature_names:
                in_v[f] = 0

        back_col, spacer_col, predict_col = st.columns([1, 4, 2])
        with back_col:
            if st.button("← Back", key="step3_back"):
                st.session_state.in_v = in_v
                st.session_state.form_step = 2
                st.rerun()
        with predict_col:
            # Validate all required inputs exist
            missing = [f for f in feature_names if f not in in_v]
            if st.button("🔍 Predict Now", key="step3_predict"):
                if not st.session_state.patient_name.strip():
                    st.error("⚠️ Patient name is missing. Please go back to Step 1.")
                elif missing:
                    st.error(f"⚠️ Some fields are missing: {missing}. Please review all steps.")
                else:
                    st.session_state.in_v = in_v
                    st.session_state.input_data = pd.DataFrame([[in_v[f] for f in feature_names]], columns=feature_names)
                    st.session_state.pdf_ready = False
                    st.session_state.pdf_bytes = None
                    go("Result")

# ═══════════════════════════════════════════════════════════════════
#  PAGE: RESULT
# ═══════════════════════════════════════════════════════════════════
elif st.session_state.page == "Result":

    if st.session_state.input_data is None:
        st.warning("No prediction data found. Please run a diagnosis first.")
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Go to Diagnosis", key="res_nodata_diag"):
                st.session_state.form_step = 1
                go("Dashboard")
        with col_b:
            if st.button("Home", key="res_nodata_home"):
                go("Home")
        st.stop()

    try:
        model = pickle.load(open("model/lung_model.pkl", "rb"))
        feature_names = pickle.load(open("model/features.pkl", "rb"))
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    probs  = model.predict_proba(st.session_state.input_data)[0]  # input_data is a DataFrame — no feature name warning
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

    # ── Top nav row ──
    nav_c1, nav_c2, nav_spacer = st.columns([1, 1, 6])
    with nav_c1:
        if st.button("Home", key="res_home"):
            go("Home")
    with nav_c2:
        if st.button("← New Diagnosis", key="res_back"):
            st.session_state.form_step = 1
            st.session_state.input_data = None
            st.session_state.in_v = {}
            st.session_state.pdf_ready = False
            st.session_state.pdf_bytes = None
            go("Dashboard")

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
            <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;'>
                <div>
                    <p class='section-label' style='margin-bottom:6px;'>Risk Assessment Result</p>
                    <div class='stat-number {STAT_CL[p_idx]}'>{LABELS[p_idx]}</div>
                    <p style='color:#8b949e; font-size:0.82rem; margin-top:8px;'>
                        Prediction confidence: <strong style='color:#e6edf3;'>{perc:.1f}%</strong>
                    </p>
                </div>
                <div style='text-align:right;'>
                    <p class='section-label' style='margin-bottom:6px;'>Confidence Score</p>
                    <div style='font-size:2.2rem; font-weight:700; color:#e6edf3; line-height:1;'>{perc:.1f}%</div>
                    <p style='color:#8b949e; font-size:0.78rem; margin-top:6px;'>out of 100%</p>
                </div>
            </div>
            <div class='risk-track' style='margin-top:14px;'>
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

        # ── PDF Section (lazy generation) ──
        st.markdown("""
            <div class='card' style='padding: 20px 28px;'>
                <p class='section-label'>Download Report</p>
                <p style='font-size:0.82rem; color:#8b949e; margin-bottom:16px;'>
                    Generate a detailed clinical PDF report for this patient.
                </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        def generate_pdf():
            """Generate a clean, professional PDF report."""
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            W, H = letter  # 612 x 792

            # ════════════════════════════════════════
            #  BACKGROUND — white page
            # ════════════════════════════════════════
            c.setFillColor(colors.white)
            c.rect(0, 0, W, H, fill=1, stroke=0)

            # ════════════════════════════════════════
            #  TOP ACCENT BAR
            # ════════════════════════════════════════
            c.setFillColor(colors.HexColor(COLORS[p_idx]))
            c.rect(0, H - 6, W, 6, fill=1, stroke=0)

            # ════════════════════════════════════════
            #  HEADER AREA
            # ════════════════════════════════════════
            # App name
            c.setFillColor(colors.HexColor("#1a1a2e"))
            c.setFont("Helvetica-Bold", 15)
            c.drawString(40, H - 38, "Lung Disease Prediction")

            # Report date top-right
            c.setFillColor(colors.HexColor("#6b7280"))
            c.setFont("Helvetica", 9)
            c.drawRightString(W - 40, H - 32, f"Date: {datetime.datetime.now().strftime('%d %B %Y  |  %H:%M')}")

            # Thin separator line under header
            c.setStrokeColor(colors.HexColor("#e5e7eb"))
            c.setLineWidth(1)
            c.line(40, H - 52, W - 40, H - 52)

            # ════════════════════════════════════════
            #  RESULT PILL BADGE
            # ════════════════════════════════════════
            badge_x, badge_y = 40, H - 98
            badge_w, badge_h = 200, 34
            c.setFillColor(colors.HexColor(COLORS[p_idx]))
            c.roundRect(badge_x, badge_y, badge_w, badge_h, 6, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 14)
            label_text = LABELS[p_idx]
            text_w = c.stringWidth(label_text, "Helvetica-Bold", 14)
            c.drawString(badge_x + (badge_w - text_w) / 2, badge_y + 11, label_text)

            # Confidence score to the right of badge
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica-Bold", 22)
            c.drawString(260, H - 90, f"{perc:.1f}%")
            c.setFillColor(colors.HexColor("#6b7280"))
            c.setFont("Helvetica", 9)
            c.drawString(260, H - 102, "Confidence Score")

            # Confidence progress bar
            bar_x, bar_y_pos, bar_w_full, bar_h = 40, H - 114, W - 80, 7
            c.setFillColor(colors.HexColor("#e5e7eb"))
            c.roundRect(bar_x, bar_y_pos, bar_w_full, bar_h, 3, fill=1, stroke=0)
            fill_w = bar_w_full * (perc / 100)
            c.setFillColor(colors.HexColor(COLORS[p_idx]))
            c.roundRect(bar_x, bar_y_pos, fill_w, bar_h, 3, fill=1, stroke=0)

            # ════════════════════════════════════════
            #  PATIENT INFO CARD  (light grey box)
            # ════════════════════════════════════════
            card_y = H - 190
            c.setFillColor(colors.HexColor("#f9fafb"))
            c.roundRect(40, card_y, W - 80, 58, 6, fill=1, stroke=0)
            c.setStrokeColor(colors.HexColor("#e5e7eb"))
            c.setLineWidth(0.8)
            c.roundRect(40, card_y, W - 80, 58, 6, fill=0, stroke=1)

            # Left column: Patient Name
            c.setFillColor(colors.HexColor("#9ca3af"))
            c.setFont("Helvetica", 7.5)
            c.drawString(56, card_y + 44, "PATIENT NAME")
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(56, card_y + 28, st.session_state.patient_name or "N/A")

            # Middle column: Report Date
            c.setFillColor(colors.HexColor("#9ca3af"))
            c.setFont("Helvetica", 7.5)
            c.drawString(230, card_y + 44, "REPORT DATE")
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(230, card_y + 28, datetime.datetime.now().strftime("%d %B %Y"))

            # Right column: Risk
            c.setFillColor(colors.HexColor("#9ca3af"))
            c.setFont("Helvetica", 7.5)
            c.drawString(420, card_y + 44, "RISK LEVEL")
            c.setFillColor(colors.HexColor(COLORS[p_idx]))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(420, card_y + 28, LABELS[p_idx])

            # Small dividers between columns
            c.setStrokeColor(colors.HexColor("#e5e7eb"))
            c.setLineWidth(0.6)
            c.line(218, card_y + 10, 218, card_y + 50)
            c.line(408, card_y + 10, 408, card_y + 50)

            # Leave extra info below the card
            c.setFillColor(colors.HexColor("#9ca3af"))
            c.setFont("Helvetica", 7.5)
            c.drawString(56, card_y + 10, "CONFIDENCE")
            c.setFillColor(colors.HexColor("#111827"))
            c.setFont("Helvetica-Bold", 9)
            c.drawString(110, card_y + 10, f"{perc:.1f}%")

            # ════════════════════════════════════════
            #  HELPER: draw a section heading
            # ════════════════════════════════════════
            def section_heading(text, y_pos, color="#111827"):
                c.setFillColor(colors.HexColor(color))
                c.setFont("Helvetica-Bold", 10)
                c.drawString(40, y_pos, text.upper())
                c.setStrokeColor(colors.HexColor(color))
                c.setLineWidth(1.5)
                c.line(40, y_pos - 5, W - 40, y_pos - 5)
                return y_pos - 20

            # ════════════════════════════════════════
            #  CLINICAL SUMMARY
            # ════════════════════════════════════════
            y = H - 218
            y = section_heading("Clinical Summary", y, "#1d4ed8")
            c.setFillColor(colors.HexColor("#374151"))
            c.setFont("Helvetica", 10)
            summary_words = msg_map[p_idx].split()
            line = ""
            for word in summary_words:
                test = line + " " + word if line else word
                if c.stringWidth(test, "Helvetica", 10) < W - 100:
                    line = test
                else:
                    c.drawString(56, y, line)
                    y -= 15
                    line = word
            if line:
                c.drawString(56, y, line)
                y -= 22

            # ════════════════════════════════════════
            #  RECOMMENDED ACTIONS
            # ════════════════════════════════════════
            y = section_heading("Recommended Actions", y, "#15803d")
            for item in do_map[p_idx]:
                # Green dot bullet
                c.setFillColor(colors.HexColor("#16a34a"))
                c.circle(52, y + 3, 3, fill=1, stroke=0)
                c.setFillColor(colors.HexColor("#374151"))
                c.setFont("Helvetica", 10)
                c.drawString(62, y, item)
                y -= 17
            y -= 6

            # ════════════════════════════════════════
            #  WHAT TO AVOID
            # ════════════════════════════════════════
            y = section_heading("What To Avoid", y, "#b91c1c")
            for item in dont_map[p_idx]:
                # Red dot bullet
                c.setFillColor(colors.HexColor("#dc2626"))
                c.circle(52, y + 3, 3, fill=1, stroke=0)
                c.setFillColor(colors.HexColor("#374151"))
                c.setFont("Helvetica", 10)
                c.drawString(62, y, item)
                y -= 17
            y -= 6

            # ════════════════════════════════════════
            #  PROBABILITY BREAKDOWN
            # ════════════════════════════════════════
            y = section_heading("Probability Breakdown", y, "#92400e")
            class_labels = {0: ("High Risk", "#ef4444"), 1: ("Low Risk", "#22c55e"), 2: ("Moderate Risk", "#f59e0b")}
            for idx, (label, bar_color) in class_labels.items():
                prob_pct = probs[idx] * 100
                c.setFillColor(colors.HexColor("#374151"))
                c.setFont("Helvetica", 9)
                c.drawString(56, y, label)
                c.setFont("Helvetica-Bold", 9)
                c.drawRightString(W - 40, y, f"{prob_pct:.1f}%")
                y -= 12
                # Track
                c.setFillColor(colors.HexColor("#e5e7eb"))
                c.roundRect(56, y, W - 112, 7, 3, fill=1, stroke=0)
                # Fill
                bar_w = (W - 112) * (prob_pct / 100)
                if bar_w > 0:
                    c.setFillColor(colors.HexColor(bar_color))
                    c.roundRect(56, y, bar_w, 7, 3, fill=1, stroke=0)
                y -= 18

            # ════════════════════════════════════════
            #  BOTTOM ACCENT LINE (matches top)
            # ════════════════════════════════════════
            c.setFillColor(colors.HexColor(COLORS[p_idx]))
            c.rect(0, 0, W, 5, fill=1, stroke=0)

            c.save()
            buf.seek(0)
            return buf.read()

        # Show Generate button first; on click generate and cache
        if not st.session_state.pdf_ready:
            if st.button("📄 Generate PDF Report", key="gen_pdf"):
                with st.spinner("Generating report..."):
                    st.session_state.pdf_bytes = generate_pdf()
                    st.session_state.pdf_ready = True
                st.rerun()
        else:
            fname = f"lung_report_{(st.session_state.patient_name or 'patient').replace(' ', '_')}.pdf"
            st.download_button(
                label="⬇️ Download PDF Report",
                data=st.session_state.pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                key="download_pdf"
            )
            if st.button("🔄 Regenerate PDF", key="regen_pdf"):
                st.session_state.pdf_ready = False
                st.session_state.pdf_bytes = None
                st.rerun()

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
