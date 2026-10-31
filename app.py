import html
import os

import joblib
import streamlit as st
from huggingface_hub import hf_hub_download


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="HealthAI | Symptoms Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Your model on the Hugging Face Hub (scikit-learn pipeline saved with joblib)
REPO_ID = "Muh340/healthcare-symptoms-disease-classifier"
MODEL_FILENAME = "healthcare_disease_model.pkl"

# Only needed if the model repo is private. Add it as a secret/env variable.
HF_TOKEN = os.environ.get("HF_TOKEN") or None


# ----------------------------------
# CUSTOM DESIGN
# ----------------------------------

st.markdown(
    """
<style>

.stApp {
    background: linear-gradient(135deg, #eff6ff, #f0fdfa, #f5f3ff);
}

.main-title {
    background: linear-gradient(90deg, #2563eb, #0d9488, #7c3aed);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.main-title h1 {
    color: white;
    font-size: 38px;
    font-weight: 800;
}

.main-title p {
    color: white;
    font-size: 17px;
}

.info-card {
    background: white;
    color: #1f2937;
    padding: 22px;
    border-radius: 15px;
    border-left: 5px solid #0d9488;
    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

.result-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #0d9488;
    box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    text-align: center;
}

.result-card h1 {
    color: #111827;
}

.result-card h2 {
    color: #0f766e;
}

.stButton button {
    background: linear-gradient(90deg, #2563eb, #0d9488);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton button:hover {
    color: white;
    border: 1px solid #2563eb;
}

</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------
# MODEL LOADING
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


# ----------------------------------
# PREDICTION
# ----------------------------------

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
    """Callback for the Clear button (runs before widgets are re-drawn)."""
    st.session_state["symptoms_input"] = ""


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:
    st.title("🏥 HealthAI")
    st.write("AI Healthcare Classification")
    st.divider()

    st.subheader("Navigation")
    page = st.radio("Choose a page", ["🏠 Home", "ℹ️ About", "📋 Disclaimer"])

    st.divider()
    st.caption("Scikit-learn model hosted on Hugging Face")
    st.caption("Educational project")


# ----------------------------------
# HOME PAGE
# ----------------------------------

if page == "🏠 Home":

    st.markdown(
        """
    <div class="main-title">
        <h1>🏥 HealthAI Classifier</h1>
        <p>Machine Learning Based Symptoms Classification</p>
        <p>Describe symptoms and explore the model's predicted category.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🌲 Random Forest\n\nTF-IDF + scikit-learn")
    with col2:
        st.success("⚡ Fast Processing\n\nAutomated text classification")
    with col3:
        st.warning("📊 Prediction\n\nModel-generated category")

    st.write("")
    st.markdown("## 📝 Enter Your Symptoms")

    st.markdown(
        """
    <div class="info-card">
        <b>How to use this application?</b><br>
        List your symptoms separated by commas, for example:
        <i>fever, sore throat, shortness of breath</i>.
        Then click the Analyze button to see the model output.
    </div>
    """,
        unsafe_allow_html=True,
    )

    symptoms = st.text_area(
        "Describe your symptoms:",
        key="symptoms_input",
        placeholder="Example: fever, headache, cough",
        height=150,
        max_chars=2000,
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        predict_button = st.button("🔍 Analyze Symptoms", use_container_width=True)

    with col2:
        st.button("🗑️ Clear", on_click=clear_text, use_container_width=True)

    if predict_button:

        if not symptoms.strip():
            st.error("Please enter your symptoms first.")

        else:
            try:
                model = load_model()

                if not has_known_words(model, symptoms):
                    st.warning(
                        "None of the words you entered match the model's "
                        "vocabulary (check spelling), so any result below "
                        "would be arbitrary. Try common symptom names such as "
                        "'fever', 'cough' or 'back pain'."
                    )
                    st.stop()

                with st.spinner("🤖 Analyzing your text..."):
                    prediction, confidence, classes, probabilities = predict_symptoms(
                        symptoms, model
                    )

                st.markdown("---")
                st.markdown("## 📊 Classification Result")

                st.markdown(
                    f"""
                <div class="result-card">
                    <h2>Predicted Category</h2>
                    <h1>{html.escape(str(prediction))}</h1>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                st.write("")
                st.subheader("Model Confidence")
                st.progress(min(max(confidence, 0.0), 1.0))
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")

                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not the probability that you have a disease."
                )

                st.subheader("Top 5 Class Probabilities")

                results = sorted(
                    zip(classes, probabilities.tolist()),
                    key=lambda x: x[1],
                    reverse=True,
                )

                for label, score in results[:5]:
                    st.write(f"**{label}** — {score * 100:.2f}%")
                    st.progress(min(max(score, 0.0), 1.0))

                st.warning(
                    "Reliability note: on the held-out test set this model "
                    "scored about 3.6% accuracy across 30 diseases, which is "
                    "close to random guessing (about 3.3%). Treat the output "
                    "as a demo only."
                )

                st.info(
                    "This is an experimental AI classification result. "
                    "It is not a diagnosis or treatment recommendation."
                )

            except Exception as e:
                st.error("Unable to load or run the model.")
                st.exception(e)


# ----------------------------------
# ABOUT PAGE
# ----------------------------------

elif page == "ℹ️ About":

    st.markdown("## 💡 About HealthAI")

    st.markdown(
        """
    HealthAI is an educational machine learning project.

    It uses a scikit-learn pipeline (TF-IDF text features followed by a
    Random Forest classifier) trained on the Kaggle
    *Healthcare Symptoms-Disease Classification* dataset to predict one of
    30 disease categories from a text list of symptoms.

    **Technologies used:**

    - Python
    - Streamlit
    - scikit-learn
    - Hugging Face Hub
    """
    )

    st.warning(
        "On its test split (5,000 samples, 30 classes) the model reached "
        "about 3.6% accuracy, roughly the level of random guessing. "
        "The dataset's symptoms appear to carry little real signal about "
        "the disease label, so this app is a deployment demo rather than "
        "a useful predictor."
    )

    st.markdown(
        f"[View trained model on Hugging Face](https://huggingface.co/{REPO_ID})"
    )


# ----------------------------------
# DISCLAIMER PAGE
# ----------------------------------

elif page == "📋 Disclaimer":

    st.title("⚠️ Medical Disclaimer")

    st.warning(
        """
    This application is developed for educational and
    research purposes only.

    It does not provide medical diagnosis, treatment,
    or professional healthcare advice.

    Model predictions may be incorrect or incomplete.

    Always consult a qualified healthcare professional
    for medical concerns.

    Do not use this application for emergency decisions.
    """
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.markdown(
    "<center>💙 HealthAI | Built with Python, Streamlit & scikit-learn</center>",
    unsafe_allow_html=True,
)
