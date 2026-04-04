
# =============================================================================
#  FILE: requirements.txt
# =============================================================================
"""
pandas
numpy
scikit-learn
streamlit
matplotlib
reportlab
"""

# =============================================================================
#  FILE: train_model.py
# =============================================================================
"""
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

# Try XGBoost (if installed)
try:
    from xgboost import XGBClassifier
    use_xgb = True
except:
    use_xgb = False

# -------------------- LOAD DATA --------------------
df = pd.read_csv("dataset/lung_cancer.csv")

print("Dataset Loaded Successfully")
print(df.head())

# -------------------- DATA CLEANING --------------------

# Remove unwanted column if exists
if "index" in df.columns:
    df.drop("index", axis=1, inplace=True)

# Encode categorical columns (Yes/No -> 1/0)
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -------------------- FEATURE & TARGET --------------------

target_column = "LUNG_CANCER"   # change if your dataset uses 'Level'

X = df.drop(target_column, axis=1)
y = df[target_column]

# Save feature names (VERY IMPORTANT for app)
feature_names = X.columns.tolist()

# -------------------- TRAIN TEST SPLIT --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------- MODEL --------------------

if use_xgb:
    print("Using XGBoost Model")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        eval_metric='logloss'
    )
else:
    print("Using Random Forest Model")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10
    )

# Train
model.fit(X_train, y_train)

# -------------------- EVALUATION --------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -------------------- SAVE MODEL --------------------

pickle.dump(model, open("model/lung_model.pkl", "wb"))

# Save feature names
pickle.dump(feature_names, open("model/features.pkl", "wb"))

print("Model saved successfully!")
"""

# =============================================================================
#  FILE: check_classes.py
# =============================================================================
"""
import pickle
model = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/lung_model.pkl", "rb"))
print("Classes:", model.classes_)
"""

# =============================================================================
#  FILE: inspect_model.py
# =============================================================================
"""
import pickle
import numpy as np

model = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/lung_model.pkl", "rb"))

print("Number of features:", model.n_features_in_)

# Check some thresholds in the first tree to see the range of values
tree = model.estimators_[0].tree_
thresholds = tree.threshold[tree.threshold != -2]  # -2 is for leaves
print("Sample Thresholds (Min/Max):", np.min(thresholds), np.max(thresholds))

# If thresholds are around 1, 2, 3, then it's a 1-8 or 1-3 scale.
# If they are around 0.5, then it's a 0/1 binary scale.
"""

# =============================================================================
#  FILE: test_risk_mapping.py
# =============================================================================
"""
import pickle
import numpy as np

model = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/lung_model.pkl", "rb"))
features = pickle.load(open("C:/Users/Admin/Desktop/Lung02/model/features.pkl", "rb"))

# High risk dummy
high_risk = np.array([[8]*25])
for i, f in enumerate(features):
    if f == "Age": high_risk[0][i] = 60
    if f == "Gender": high_risk[0][i] = 1  # Male
    if f in ["index", "Patient Id"]: high_risk[0][i] = 0

p_high = model.predict_proba(high_risk)[0]
print("High Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_high):
    print(i, prob)

# Low risk dummy
low_risk = np.array([[1]*25])
for i, f in enumerate(features):
    if f == "Age": low_risk[0][i] = 20
    if f == "Gender": low_risk[0][i] = 2  # Female
    if f in ["index", "Patient Id"]: low_risk[0][i] = 0

p_low = model.predict_proba(low_risk)[0]
print("\nLow Risk input results (Classes 0, 1, 2):")
for i, prob in enumerate(p_low):
    print(i, prob)
"""

# =============================================================================
#  FILE: save_features.py
# =============================================================================
"""
(Empty file - no code)
"""

# =============================================================================
#  FILE: app.py  (MAIN STREAMLIT APPLICATION)
# =============================================================================

import streamlit as st
import pickle
import numpy as np
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

/* ── Step indicator ── */
.step-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
}
.step-item {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}
.step-circle {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
}
.step-active  { background: #238636; color: white; border: 2px solid #2ea043; }
.step-done    { background: #1f6feb; color: white; border: 2px solid #58a6ff; }
.step-pending { background: #21262d; color: #8b949e; border: 2px solid #30363d; }
.step-label {
    font-size: 0.8rem;
    font-weight: 500;
}
.step-label-active  { color: #e6edf3; }
.step-label-done    { color: #58a6ff; }
.step-label-pending { color: #8b949e; }
.step-line {
    flex: 1;
    height: 2px;
    background: #30363d;
    margin: 0 8px;
}
.step-line-done { background: #3fb950; }

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

# ─────────────────────────── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🫁 Lung Disease Prediction</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.78rem; color:#8b949e; margin-bottom:1.5rem;'>Clinical diagnostic tool powered by Random Forest classification.</p>", unsafe_allow_html=True)

    nav_items = [("🏠  Home", "Home"), ("🩺  Run Diagnosis", "Dashboard"), ("📋  View Result", "Result")]
    for label, target in nav_items:
        active = st.session_state.page == target
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

    step_html = "<div class='step-bar'>"
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
            line_cls = "step-line-done" if i < step else ""
            step_html += f"<div class='step-line {line_cls}'></div>"
    step_html += "</div>"
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
            placeholder="e.g. Ramesh Kumar"
        )
        st.session_state.patient_name = patient_name

        c1, c2, c3 = st.columns(3)
        with c1:
            age_val = in_v.get("Age", 1)
            in_v["Age"] = st.slider("Age *", 1, 100, age_val)
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
        st.markdown("<p style='font-size:0.78rem; color:#8b949e;'>* Required fields</p>", unsafe_allow_html=True)

        _, btn_col = st.columns([6, 1])
        with btn_col:
            if st.button("Next →", key="step1_next"):
                if not patient_name.strip():
                    st.error("⚠️ Please enter the patient's full name before proceeding.")
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
                    st.session_state.input_data = np.array([[in_v[f] for f in feature_names]])
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
            if st.button("🔬 Go to Diagnosis", key="res_nodata_diag"):
                st.session_state.form_step = 1
                go("Dashboard")
        with col_b:
            if st.button("🏠 Home", key="res_nodata_home"):
                go("Home")
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

    # ── Top nav row ──
    nav_c1, nav_c2, nav_spacer = st.columns([1, 1, 6])
    with nav_c1:
        if st.button("🏠 Home", key="res_home"):
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
            """Generate a rich, well-formatted PDF report."""
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=letter)
            W, H = letter  # 612 x 792

            # ── Header Banner ──
            c.setFillColor(colors.HexColor("#0d1117"))
            c.rect(0, H - 80, W, 80, fill=1, stroke=0)

            # Logo text + title
            c.setFillColor(colors.HexColor("#58a6ff"))
            c.setFont("Helvetica-Bold", 10)
            c.drawString(40, H - 28, "🫁  LUNG DISEASE PREDICTION")

            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 18)
            c.drawString(40, H - 52, "Clinical Diagnostic Report")

            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#8b949e"))
            c.drawRightString(W - 40, H - 28, f"Generated: {datetime.datetime.now().strftime('%d %b %Y  %H:%M')}")
            c.drawRightString(W - 40, H - 44, "For reference use only · Consult a physician")

            # ── Result Banner ──
            result_color = COLORS[p_idx]
            c.setFillColor(colors.HexColor(result_color))
            c.rect(0, H - 130, W, 48, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(40, H - 112, f"RESULT:  {LABELS[p_idx]}")
            c.setFont("Helvetica", 13)
            c.drawRightString(W - 40, H - 112, f"Confidence:  {perc:.1f}%")

            # ── Progress bar ──
            bar_y = H - 148
            c.setFillColor(colors.HexColor("#21262d"))
            c.roundRect(40, bar_y, W - 80, 10, 5, fill=1, stroke=0)
            fill_w = (W - 80) * (perc / 100)
            c.setFillColor(colors.HexColor(result_color))
            c.roundRect(40, bar_y, fill_w, 10, 5, fill=1, stroke=0)

            # ── Patient Info Box ──
            c.setFillColor(colors.HexColor("#161b22"))
            c.roundRect(40, H - 230, W - 80, 68, 6, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#30363d"))
            c.roundRect(40, H - 230, W - 80, 68, 6, fill=0, stroke=1)

            c.setFillColor(colors.HexColor("#8b949e"))
            c.setFont("Helvetica", 8)
            c.drawString(56, H - 172, "PATIENT NAME")
            c.drawString(280, H - 172, "REPORT DATE")

            c.setFillColor(colors.HexColor("#e6edf3"))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(56, H - 190, st.session_state.patient_name or "N/A")
            c.drawString(280, H - 190, datetime.datetime.now().strftime("%d %B %Y"))

            c.setFillColor(colors.HexColor("#8b949e"))
            c.setFont("Helvetica", 8)
            c.drawString(56, H - 210, "RISK CATEGORY")
            c.drawString(280, H - 210, "CONFIDENCE SCORE")

            c.setFillColor(colors.HexColor(result_color))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(56, H - 225, LABELS[p_idx])
            c.drawString(280, H - 225, f"{perc:.1f}%")

            # ── Section: Clinical Summary ──
            y = H - 260
            c.setFillColor(colors.HexColor("#58a6ff"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "■  CLINICAL SUMMARY")
            c.setFillColor(colors.HexColor("#30363d"))
            c.rect(40, y - 4, W - 80, 1, fill=1, stroke=0)

            y -= 20
            c.setFillColor(colors.HexColor("#c9d1d9"))
            c.setFont("Helvetica", 10)
            # Word-wrap the summary text
            summary_words = msg_map[p_idx].split()
            line = ""
            for word in summary_words:
                test = line + " " + word if line else word
                if c.stringWidth(test, "Helvetica", 10) < W - 100:
                    line = test
                else:
                    c.drawString(56, y, line)
                    y -= 16
                    line = word
            if line:
                c.drawString(56, y, line)
                y -= 24

            # ── Section: Recommendations ──
            c.setFillColor(colors.HexColor("#3fb950"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "■  RECOMMENDED ACTIONS")
            c.setFillColor(colors.HexColor("#30363d"))
            c.rect(40, y - 4, W - 80, 1, fill=1, stroke=0)
            y -= 18

            for item in do_map[p_idx]:
                c.setFillColor(colors.HexColor("#3fb950"))
                c.setFont("Helvetica-Bold", 10)
                c.drawString(56, y, "✓")
                c.setFillColor(colors.HexColor("#c9d1d9"))
                c.setFont("Helvetica", 10)
                c.drawString(72, y, item)
                y -= 18
            y -= 8

            # ── Section: Restrictions ──
            c.setFillColor(colors.HexColor("#f85149"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "■  RESTRICTIONS / WHAT TO AVOID")
            c.setFillColor(colors.HexColor("#30363d"))
            c.rect(40, y - 4, W - 80, 1, fill=1, stroke=0)
            y -= 18

            for item in dont_map[p_idx]:
                c.setFillColor(colors.HexColor("#f85149"))
                c.setFont("Helvetica-Bold", 10)
                c.drawString(56, y, "✗")
                c.setFillColor(colors.HexColor("#c9d1d9"))
                c.setFont("Helvetica", 10)
                c.drawString(72, y, item)
                y -= 18
            y -= 12

            # ── Section: All Class Probabilities ──
            c.setFillColor(colors.HexColor("#e3b341"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(40, y, "■  PROBABILITY BREAKDOWN")
            c.setFillColor(colors.HexColor("#30363d"))
            c.rect(40, y - 4, W - 80, 1, fill=1, stroke=0)
            y -= 20

            class_labels = {0: ("High Risk", "#f85149"), 1: ("Low Risk", "#3fb950"), 2: ("Moderate Risk", "#e3b341")}
            for idx, (label, bar_color) in class_labels.items():
                prob_pct = probs[idx] * 100
                c.setFillColor(colors.HexColor("#8b949e"))
                c.setFont("Helvetica", 9)
                c.drawString(56, y, label)
                c.drawRightString(W - 40, y, f"{prob_pct:.1f}%")
                y -= 14
                # Bar background
                c.setFillColor(colors.HexColor("#21262d"))
                c.roundRect(56, y, W - 112, 8, 4, fill=1, stroke=0)
                # Bar fill
                bar_w = (W - 112) * (prob_pct / 100)
                if bar_w > 0:
                    c.setFillColor(colors.HexColor(bar_color))
                    c.roundRect(56, y, bar_w, 8, 4, fill=1, stroke=0)
                y -= 20

            # ── Footer ──
            c.setFillColor(colors.HexColor("#161b22"))
            c.rect(0, 0, W, 48, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#8b949e"))
            c.setFont("Helvetica-Oblique", 8)
            c.drawString(40, 30, "Lung Disease Prediction  ·  Clinical Suite v3.8  ·  For research & reference use only")
            c.drawRightString(W - 40, 30, "This report does NOT replace a clinical diagnosis. Consult a certified physician.")

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

# ── Footer ──
st.markdown("""
    <div style='margin-top:3rem; padding-top:1.5rem; border-top:1px solid #21262d;
                text-align:center; font-size:0.72rem; color:#484f58;'>
        Lung Disease Prediction &nbsp;·&nbsp; Clinical Suite v3.8 &nbsp;·&nbsp; For research use only
    </div>
""", unsafe_allow_html=True)
