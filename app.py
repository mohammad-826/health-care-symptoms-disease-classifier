
import base64
import html
import os
from pathlib import Path
 
import joblib
import streamlit as st
from huggingface_hub import hf_hub_download
 
 
# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
 
APP_TITLE = "Health-Care Symptoms Classifier"
DEVELOPER_NAME = "Muhammad"
GITHUB_USER = "mohammad-826"  # change if your GitHub username is different
GITHUB_URL = f"https://github.com/{GITHUB_USER}"
HF_PROFILE_URL = "https://huggingface.co/Muh340"
 
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# Your model on the Hugging Face Hub (scikit-learn pipeline saved with joblib)
REPO_ID = "Muh340/healthcare-symptoms-disease-classifier"
MODEL_FILENAME = "healthcare_disease_model.pkl"
 
# Only needed if the model repo is private. Add it as a secret/env variable.
HF_TOKEN = os.environ.get("HF_TOKEN") or None
 
APP_DIR = Path(__file__).parent
 
# Optional hero photo: put your own image at assets/hero.jpg (or .png/.webp).
# If none is found, a built-in illustration is used instead.
HERO_FILES = ["hero.jpg", "hero.jpeg", "hero.png", "hero.webp"]
 
EXAMPLES = [
    "fever, back pain, shortness of breath",
    "insomnia, back pain, weight loss",
    "sore throat, vomiting, diarrhea",
]
 
 
# ----------------------------------
# CUSTOM DESIGN
# ----------------------------------
 
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700;800&display=swap');
 
html { scroll-behavior: smooth; }
 
.stApp { background: #eef2f7; }
 
header[data-testid="stHeader"], #MainMenu, footer,
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stToolbar"] {
    display: none !important;
}
 
.block-container {
    max-width: 1180px;
    padding: 1.2rem 1.5rem 3rem;
}
 
.hc, .hc * {
    font-family: 'Work Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
    box-sizing: border-box;
}
 
.stApp textarea, .stApp label p, .stApp button p {
    font-family: 'Work Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
 
/* ---------- NAV ---------- */
.hc .nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 8px 6px 20px;
}
.hc .brand {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; font-size: 20px; color: #0f1b3d;
}
.hc .brand-logo {
    width: 36px; height: 36px; border-radius: 50%;
    background: #0b4fff; color: #fff;
    display: flex; align-items: center; justify-content: center; font-size: 17px;
}
.hc .nav-links { display: flex; gap: 30px; }
.hc .nav-links a {
    color: #475569; text-decoration: none; font-weight: 500; font-size: 15px;
}
.hc .nav-links a:hover { color: #0b4fff; }
 
/* ---------- BUTTON LINKS ---------- */
.hc .btn {
    display: inline-block; padding: 14px 28px; border-radius: 999px;
    font-weight: 600; font-size: 15px; text-decoration: none !important;
    transition: all .2s ease; cursor: pointer;
}
.hc .btn-sm { padding: 10px 22px; font-size: 14px; }
.hc .btn-primary { background: #0b4fff; color: #fff !important; }
.hc .btn-primary:hover { background: #0840d1; }
.hc .btn-light { background: #fff; color: #0f1b3d !important; }
.hc .btn-light:hover { box-shadow: 0 8px 20px rgba(15, 27, 61, .14); }
 
/* ---------- HERO ---------- */
.hc .hero {
    display: grid; grid-template-columns: 1.05fr .95fr;
    border-radius: 22px; overflow: hidden; min-height: 400px;
    background: linear-gradient(135deg, #e4e9ef 0%, #ced7e0 100%);
}
.hc .hero-text {
    padding: 52px 24px 52px 52px;
    display: flex; flex-direction: column; justify-content: center; align-items: flex-start;
}
.hc .pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fff; color: #334155; font-size: 14px; font-weight: 500;
    padding: 8px 16px; border-radius: 999px; margin-bottom: 26px;
}
.hc .pill-icon { color: #0b4fff; font-size: 16px; }
.hc .hero h1 {
    margin: 0 0 18px !important; padding: 0 !important;
    font-size: clamp(36px, 5vw, 58px); line-height: 1.08; font-weight: 700;
    color: #0f1b3d; letter-spacing: -0.02em;
}
.hc .hero p {
    margin: 0 0 30px; max-width: 440px;
    font-size: 17px; line-height: 1.55; color: #475569;
}
.hc .hero-actions { display: flex; gap: 12px; flex-wrap: wrap; }
.hc .hero-photo { position: relative; min-height: 380px; }
.hc .hero-photo img {
    position: absolute; inset: 0; width: 100%; height: 100%;
    object-fit: cover; object-position: center top;
    -webkit-mask-image: linear-gradient(to right, transparent 0, #000 30%);
    mask-image: linear-gradient(to right, transparent 0, #000 30%);
}
.hc .hero-art {
    display: flex; align-items: center; justify-content: center; padding: 20px;
}
.hc .hero-art img { width: 100%; max-width: 520px; }
 
/* ---------- STATS ---------- */
.hc .stats {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 22px 0 8px;
}
.hc .stat {
    background: #fff; border-radius: 18px; padding: 22px 24px;
    box-shadow: 0 4px 16px rgba(15, 27, 61, .06);
}
.hc .stat b { display: block; font-size: 26px; color: #0b4fff; font-weight: 700; }
.hc .stat span { font-size: 14px; color: #64748b; }
 
/* ---------- SECTIONS ---------- */
.hc .section-head { margin: 46px 0 18px; }
.hc .eyebrow {
    display: inline-block; font-size: 13px; font-weight: 700; letter-spacing: .08em;
    text-transform: uppercase; color: #0b4fff; margin-bottom: 8px;
}
.hc .section-head h2 {
    margin: 0 0 8px !important; padding: 0 !important;
    font-size: 32px; font-weight: 700; color: #0f1b3d; letter-spacing: -0.01em;
}
.hc .section-head p { margin: 0; color: #64748b; font-size: 16px; max-width: 620px; }
 
.hc .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.hc .card {
    background: #fff; border-radius: 20px; padding: 28px;
    box-shadow: 0 4px 16px rgba(15, 27, 61, .06);
}
.hc .card .num {
    width: 40px; height: 40px; border-radius: 50%; background: #e8efff; color: #0b4fff;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; margin-bottom: 16px;
}
.hc .card h3 {
    margin: 0 0 8px !important; padding: 0 !important;
    font-size: 19px; color: #0f1b3d; font-weight: 600;
}
.hc .card p { margin: 0; color: #64748b; line-height: 1.6; font-size: 15px; }
.hc .card a { color: #0b4fff; font-weight: 600; text-decoration: none; }
 
/* ---------- INPUT WIDGETS ---------- */
.stTextArea textarea {
    border-radius: 16px !important; border: 1.5px solid #cbd5e1 !important;
    background: #fff !important; color: #0f1b3d !important;
    font-size: 16px !important; padding: 16px !important;
}
.stTextArea textarea:focus {
    border-color: #0b4fff !important; box-shadow: 0 0 0 3px rgba(11, 79, 255, .15) !important;
}
.stTextArea label p { font-weight: 600 !important; color: #0f1b3d !important; }
 
.stButton, .stButton button { width: 100%; }
.stButton button {
    border-radius: 999px; padding: .7rem 1.4rem; font-weight: 600;
    transition: all .2s ease; border: none;
}
.stButton button[kind="primary"],
.stButton button[data-testid="stBaseButton-primary"] {
    background: #0b4fff; color: #fff;
}
.stButton button[kind="primary"]:hover,
.stButton button[data-testid="stBaseButton-primary"]:hover { background: #0840d1; color: #fff; }
.stButton button[kind="primary"] p,
.stButton button[data-testid="stBaseButton-primary"] p { color: #fff !important; }
 
.stButton button[kind="secondary"],
.stButton button[data-testid="stBaseButton-secondary"] {
    background: #fff; color: #0f1b3d; border: 1px solid #dbe2ea;
}
.stButton button[kind="secondary"]:hover,
.stButton button[data-testid="stBaseButton-secondary"]:hover {
    border-color: #0b4fff; color: #0b4fff;
}
.stButton button[kind="secondary"] p,
.stButton button[data-testid="stBaseButton-secondary"] p { color: inherit !important; }
 
/* ---------- RESULTS ---------- */
.hc .result {
    background: #fff; border-radius: 22px; padding: 34px; text-align: center;
    border: 2px solid #0b4fff; box-shadow: 0 10px 30px rgba(11, 79, 255, .10);
    margin-top: 22px;
}
.hc .result .eyebrow { margin-bottom: 6px; }
.hc .result h2 {
    margin: 4px 0 14px !important; padding: 0 !important;
    font-size: 40px; font-weight: 700; color: #0f1b3d;
}
.hc .chip {
    display: inline-block; background: #e8efff; color: #0b4fff;
    font-weight: 600; font-size: 14px; padding: 7px 16px; border-radius: 999px;
}
.hc .bars {
    background: #fff; border-radius: 20px; padding: 26px 28px; margin-top: 18px;
    box-shadow: 0 4px 16px rgba(15, 27, 61, .06);
}
.hc .bars h3 {
    margin: 0 0 16px !important; padding: 0 !important;
    font-size: 18px; color: #0f1b3d; font-weight: 600;
}
.hc .bar-row { margin-bottom: 14px; }
.hc .bar-label {
    display: flex; justify-content: space-between;
    font-size: 15px; color: #334155; margin-bottom: 6px; font-weight: 500;
}
.hc .bar-track { background: #e8eef7; border-radius: 999px; height: 10px; overflow: hidden; }
.hc .bar-fill { background: linear-gradient(90deg, #0b4fff, #4f8bff); height: 100%; border-radius: 999px; }
.hc .bars small { color: #94a3b8; font-size: 12px; }
 
.hc .callout {
    border-radius: 16px; padding: 16px 20px; margin-top: 18px;
    font-size: 15px; line-height: 1.55;
}
.hc .callout-warn { background: #fff7e6; color: #8a5a00; border: 1px solid #ffe2a8; }
.hc .callout-info { background: #e8f1ff; color: #1e3a8a; border: 1px solid #c7dbff; }
 
.hc .footer {
    text-align: center; color: #64748b; font-size: 14px;
    margin-top: 56px; padding-top: 24px; border-top: 1px solid #dbe2ea;
}
 
.hc .by {
    margin-top: 26px; font-size: 14px; color: #475569;
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
}
.hc .by b { color: #0f1b3d; }
.hc .by a { color: #0b4fff; font-weight: 600; text-decoration: none; }
.hc .by a:hover { text-decoration: underline; }
.hc .result-note { margin: 0 0 16px; color: #475569; font-size: 16px; }
.hc .avatar {
    width: 56px; height: 56px; border-radius: 50%; background: #0b4fff; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; font-weight: 700; margin-bottom: 14px;
}
.hc .card-links { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
.hc .card .btn-primary { color: #fff !important; }
.hc .card .btn-light { background: #eef2f7; }
 
@media (max-width: 860px) {
    .hc .brand { font-size: 16px; }
    .hc .nav .btn { display: none; }
    .hc .hero { grid-template-columns: 1fr; }
    .hc .hero-text { padding: 36px 26px; }
    .hc .hero-photo { min-height: 280px; }
    .hc .hero-photo img { -webkit-mask-image: none; mask-image: none; }
    .hc .stats, .hc .cards { grid-template-columns: 1fr; }
    .hc .nav-links { display: none; }
}
</style>
"""
 
st.markdown(CSS, unsafe_allow_html=True)
 
 
def md(markup):
    """Render custom HTML. Blank lines/indentation are stripped so Markdown
    never mistakes the HTML for a code block."""
    cleaned = "\n".join(
        line.strip() for line in markup.strip().splitlines() if line.strip()
    )
    st.markdown(f'<div class="hc">{cleaned}</div>', unsafe_allow_html=True)
 
 
# ----------------------------------
# HERO VISUAL (photo if provided, else illustration)
# ----------------------------------
 
FALLBACK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 480">
<defs><filter id="s" x="-20%" y="-20%" width="140%" height="150%">
<feDropShadow dx="0" dy="14" stdDeviation="16" flood-color="#0f1b3d" flood-opacity="0.18"/>
</filter></defs>
<circle cx="320" cy="250" r="200" fill="#bcd0ff" opacity="0.55"/>
<circle cx="320" cy="250" r="135" fill="#0b4fff" opacity="0.10"/>
<rect x="150" y="165" width="340" height="215" rx="30" fill="#ffffff" filter="url(#s)"/>
<polyline points="185,280 245,280 272,222 310,338 345,250 368,280 455,280"
 fill="none" stroke="#0b4fff" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="185" y="196" width="90" height="12" rx="6" fill="#dbe5f5"/>
<rect x="185" y="218" width="54" height="10" rx="5" fill="#e8eef7"/>
<circle cx="470" cy="160" r="48" fill="#0b4fff"/>
<rect x="461" y="136" width="18" height="48" rx="5" fill="#ffffff"/>
<rect x="446" y="151" width="48" height="18" rx="5" fill="#ffffff"/>
<circle cx="130" cy="125" r="14" fill="#0b4fff" opacity="0.25"/>
<circle cx="520" cy="330" r="20" fill="#0b4fff" opacity="0.18"/>
<circle cx="95" cy="350" r="10" fill="#0b4fff" opacity="0.30"/>
</svg>"""
 
 
@st.cache_data(show_spinner=False)
def hero_visual():
    for name in HERO_FILES:
        path = APP_DIR / "assets" / name
        if path.exists():
            ext = path.suffix.lower().lstrip(".")
            mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}[ext]
            data = base64.b64encode(path.read_bytes()).decode()
            return (
                '<div class="hero-photo">'
                f'<img src="data:image/{mime};base64,{data}" alt="Healthcare professional">'
                "</div>"
            )
    svg = base64.b64encode(FALLBACK_SVG.encode()).decode()
    return (
        '<div class="hero-art">'
        f'<img src="data:image/svg+xml;base64,{svg}" alt="Health illustration">'
        "</div>"
    )
 
 
# ----------------------------------
# MODEL LOADING + PREDICTION
# ----------------------------------
 
@st.cache_resource(show_spinner="Downloading and loading the model (first run can take a while)...")
def load_model():
    """Download the .pkl from the Hugging Face Hub (cached) and load it."""
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        token=HF_TOKEN,
    )
    return joblib.load(model_path)
 
 
def get_classes(model):
    """Pipeline exposes classes_ from its final step; fall back just in case."""
    if hasattr(model, "classes_"):
        return list(model.classes_)
    return list(model.named_steps["classifier"].classes_)
 
 
def has_known_words(model, text):
    """True if at least one word in the text exists in the TF-IDF vocabulary."""
    try:
        return model.named_steps["tfidf"].transform([text]).nnz > 0
    except Exception:
        return True  # can't check, so don't block the prediction
 
 
def predict_symptoms(text, model):
    # The notebook trained the pipeline on the "Symptoms" text column only,
    # so it takes a list of strings.
    probabilities = model.predict_proba([text])[0]
    classes = get_classes(model)
 
    best = int(probabilities.argmax())
    return classes[best], float(probabilities[best]), classes, probabilities
 
 
def clear_text():
    st.session_state["symptoms_input"] = ""
 
 
def set_example(text):
    st.session_state["symptoms_input"] = text
 
 
# ----------------------------------
# NAVBAR + HERO
# ----------------------------------
 
md(
    f"""
<div class="nav" id="top">
  <div class="brand"><div class="brand-logo">✚</div>{APP_TITLE}</div>
  <div class="nav-links">
    <a href="#top" target="_self">Home</a>
    <a href="#checker" target="_self">Symptom Checker</a>
    <a href="#how" target="_self">How it works</a>
    <a href="#about" target="_self">About</a>
    <a href="{GITHUB_URL}" target="_blank">GitHub</a>
  </div>
  <a class="btn btn-primary btn-sm" href="#checker" target="_self">Try it now</a>
</div>
<div class="hero">
  <div class="hero-text">
    <div class="pill"><span class="pill-icon">★</span> 30 disease classes · 25,000 records</div>
    <h1>Health-Care<br>Symptoms Classifier</h1>
    <p>Describe how you feel in a few words and see which disease categories the
    model considers most likely. An educational demo, not a medical diagnosis.</p>
    <div class="hero-actions">
      <a class="btn btn-primary" href="#checker" target="_self">Check Symptoms</a>
      <a class="btn btn-light" href="#how" target="_self">Learn More</a>
    </div>
    <div class="by">Developed by <b>{DEVELOPER_NAME}</b> · <a href="{GITHUB_URL}" target="_blank">GitHub: {GITHUB_USER}</a></div>
  </div>
  {hero_visual()}
</div>
<div class="stats">
  <div class="stat"><b>30</b><span>Disease categories</span></div>
  <div class="stat"><b>25,000</b><span>Patient records in the dataset</span></div>
  <div class="stat"><b>TF-IDF + Random Forest</b><span>Model architecture</span></div>
</div>
"""
)
 
 
# ----------------------------------
# SYMPTOM CHECKER
# ----------------------------------
 
md(
    """
<div id="checker"></div>
<div class="section-head">
  <span class="eyebrow">Symptom checker</span>
  <h2>Tell us what you are experiencing</h2>
  <p>List your symptoms separated by commas, or start from one of the examples below.</p>
</div>
"""
)
 
ex_cols = st.columns(3)
for col, example in zip(ex_cols, EXAMPLES):
    with col:
        st.button(example, key=f"ex_{example}", on_click=set_example, args=(example,))
 
symptoms = st.text_area(
    "Your symptoms",
    key="symptoms_input",
    placeholder="Example: fever, headache, cough",
    height=140,
    max_chars=2000,
)
 
st.caption(
    "Tip: use common symptom names separated by commas (for example "
    "\"fever, cough, sore throat\"). Spelling matters."
)
 
btn_col1, btn_col2 = st.columns([3, 1])
with btn_col1:
    predict_button = st.button("Analyze Symptoms", type="primary")
with btn_col2:
    st.button("Clear", on_click=clear_text)
 
if predict_button:
 
    if not symptoms.strip():
        st.error("Please enter your symptoms first.")
 
    else:
        try:
            model = load_model()
 
            if not has_known_words(model, symptoms):
                md(
                    """
                <div class="callout callout-warn">
                None of the words you entered match the model's vocabulary
                (check the spelling), so any result would be arbitrary.
                Try common symptom names such as "fever", "cough" or "back pain".
                </div>
                """
                )
 
            else:
                with st.spinner("Analyzing your symptoms, one moment..."):
                    prediction, confidence, classes, probabilities = predict_symptoms(
                        symptoms, model
                    )
 
                results = sorted(
                    zip(classes, probabilities.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                top_score = results[0][1] or 1e-9
 
                bars = "".join(
                    f"""
                    <div class="bar-row">
                      <div class="bar-label"><span>{html.escape(str(label))}</span><span>{score * 100:.2f}%</span></div>
                      <div class="bar-track"><div class="bar-fill" style="width:{score / top_score * 100:.1f}%"></div></div>
                    </div>
                    """
                    for label, score in results
                )
 
                md(
                    f"""
                <div class="result">
                  <span class="eyebrow">Predicted category</span>
                  <h2>{html.escape(str(prediction))}</h2>
                  <p class="result-note">Based on the symptoms you entered, this is the model's closest match.</p>
                  <span class="chip">Model confidence: {confidence * 100:.2f}%</span>
                </div>
                <div class="bars">
                  <h3>Top 5 class probabilities</h3>
                  {bars}
                  <small>Bars are scaled relative to the top class. Confidence is the model's
                  estimated probability, not the probability that you have a disease.</small>
                </div>
                <div class="callout callout-warn">
                  <b>Reliability note:</b> on the held-out test set this model scored about 3.6%
                  accuracy across 30 diseases, close to random guessing (about 3.3%).
                  Treat the output as a demo only.
                </div>
                <div class="callout callout-info">
                  This is an experimental AI classification result. It is not a diagnosis or a
                  treatment recommendation.
                </div>
                """
                )
 
        except Exception as e:
            st.error("Unable to load or run the model.")
            st.exception(e)
 
 
# ----------------------------------
# HOW IT WORKS
# ----------------------------------
 
md(
    """
<div id="how"></div>
<div class="section-head">
  <span class="eyebrow">How it works</span>
  <h2>From symptoms to a predicted category</h2>
  <p>Three simple steps, powered by a scikit-learn pipeline hosted on Hugging Face.</p>
</div>
<div class="cards">
  <div class="card"><div class="num">1</div><h3>Describe symptoms</h3>
    <p>Type your symptoms as a comma-separated list, such as "fever, sore throat, vomiting".</p></div>
  <div class="card"><div class="num">2</div><h3>Model analysis</h3>
    <p>TF-IDF turns your text into numeric features and a Random Forest scores all 30 disease classes.</p></div>
  <div class="card"><div class="num">3</div><h3>See the results</h3>
    <p>You get the top predicted category plus the five most likely classes with their probabilities.</p></div>
</div>
"""
)
 
 
# ----------------------------------
# ABOUT + DISCLAIMER
# ----------------------------------
 
md(
    f"""
<div id="about"></div>
<div class="section-head">
  <span class="eyebrow">About</span>
  <h2>About this project</h2>
</div>
<div class="cards">
  <div class="card"><h3>The model</h3>
    <p>{APP_TITLE} is an educational machine learning project. It uses a TF-IDF + Random Forest
    pipeline trained on the Kaggle <i>Healthcare Symptoms-Disease Classification</i> dataset
    (25,000 records, 30 diseases). Built with Python, Streamlit, scikit-learn and the
    Hugging Face Hub.<br><br>
    <a href="https://huggingface.co/{REPO_ID}" target="_blank">View the model on Hugging Face →</a></p></div>
  <div class="card"><div class="avatar">{DEVELOPER_NAME[0]}</div><h3>Developed by {DEVELOPER_NAME}</h3>
    <p>Built as a machine learning and deployment project. Source code and other projects are on GitHub.</p>
    <div class="card-links">
      <a class="btn btn-primary btn-sm" href="{GITHUB_URL}" target="_blank">GitHub</a>
      <a class="btn btn-light btn-sm" href="{HF_PROFILE_URL}" target="_blank">Hugging Face</a>
    </div></div>
  <div class="card"><h3>⚠️ Medical disclaimer</h3>
    <p>This application is for educational and research purposes only. It does not provide
    medical diagnosis, treatment or professional healthcare advice, and its predictions may be
    incorrect. Always consult a qualified healthcare professional for medical concerns, and do
    not use this application for emergency decisions.</p></div>
</div>
<div class="callout callout-warn">
  <b>Accuracy note:</b> on its test split (5,000 samples, 30 classes) the model reached about 3.6%
  accuracy, roughly the level of random guessing. The dataset's symptoms appear to carry little
  real signal about the disease label, so this app is a deployment demo rather than a useful predictor.
</div>
<div class="footer">💙 {APP_TITLE} · Developed by {DEVELOPER_NAME} · <a href="{GITHUB_URL}" target="_blank" style="color:#0b4fff;text-decoration:none;font-weight:600">GitHub</a><br>Built with Python, Streamlit &amp; scikit-learn · Educational use only</div>
"""
)
 


