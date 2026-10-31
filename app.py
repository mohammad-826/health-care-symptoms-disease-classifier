
import html
import os
 
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
 
# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------
 
st.set_page_config(
    page_title="HealthAI | Symptoms Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# Replace with YOUR model repo id if you are deploying your own model
MODEL_ID = "Muh340/healthcare-symptoms-disease-classifier"
 
# Optional: only needed if your model repo is private/gated.
# Set HF_TOKEN in Spaces -> Settings -> Secrets (or Streamlit Cloud secrets).
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
 
@st.cache_resource(show_spinner="Loading your AI model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, token=HF_TOKEN
    )
    model.eval()
    return tokenizer, model
 
 
# ----------------------------------
# HELPERS
# ----------------------------------
 
def get_label(labels, idx):
    """id2label keys can be int or str depending on how the config was saved."""
    return labels.get(idx, labels.get(str(idx), f"Class {idx}"))
 
 
def predict_symptoms(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
 
    # Some tokenizers return token_type_ids that certain models don't accept
    if "token_type_ids" in inputs and not getattr(
        model.config, "type_vocab_size", 0
    ):
        inputs.pop("token_type_ids")
 
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
 
    predicted_id = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0][predicted_id].item()
    predicted_label = get_label(model.config.id2label, predicted_id)
 
    return predicted_label, confidence, probabilities[0]
 
 
def clear_text():
    """Callback for the Clear button (runs before the widgets are re-drawn)."""
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
    st.caption("Powered by Hugging Face Transformers")
    st.caption("Educational project")
 
 
# ----------------------------------
# HOME PAGE
# ----------------------------------
 
if page == "🏠 Home":
 
    st.markdown(
        """
    <div class="main-title">
        <h1>🏥 HealthAI Classifier</h1>
        <p>Artificial Intelligence Based Symptoms Classification</p>
        <p>Describe symptoms and explore the model's predicted category.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
 
    col1, col2, col3 = st.columns(3)
 
    with col1:
        st.info("🤖 AI Powered\n\nTransformer-based model")
    with col2:
        st.success("⚡ Smart Processing\n\nAutomated text classification")
    with col3:
        st.warning("📊 Prediction\n\nModel-generated category")
 
    st.write("")
    st.markdown("## 📝 Enter Your Symptoms")
 
    st.markdown(
        """
    <div class="info-card">
        <b>How to use this application?</b><br>
        Describe your symptoms using clear and simple sentences.
        Then click the Analyze button to see the model output.
    </div>
    """,
        unsafe_allow_html=True,
    )
 
    # The key lets the Clear callback reset the text box
    symptoms = st.text_area(
        "Describe your symptoms:",
        key="symptoms_input",
        placeholder="Example: I have a headache and feel tired...",
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
                with st.spinner("🤖 AI is processing your text..."):
                    tokenizer, model = load_model()
                    prediction, confidence, probabilities = predict_symptoms(
                        symptoms, tokenizer, model
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
                st.progress(float(confidence))
                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
 
                st.caption(
                    "Confidence is the model's estimated probability, "
                    "not the probability that you have a disease."
                )
 
                st.subheader("Class Probabilities")
 
                labels = model.config.id2label
                results = [
                    (get_label(labels, i), score)
                    for i, score in enumerate(probabilities.tolist())
                ]
                results.sort(key=lambda x: x[1], reverse=True)
 
                for label, score in results[:5]:
                    st.write(f"**{label}** — {score * 100:.2f}%")
                    st.progress(float(score))
 
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
 
    It uses a trained Hugging Face model to classify
    text descriptions of symptoms into predicted categories.
 
    **Technologies used:**
 
    - Python
    - Streamlit
    - PyTorch
    - Hugging Face Transformers
    """
    )
 
    st.markdown(
        f"[View trained model on Hugging Face](https://huggingface.co/{MODEL_ID})"
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
    "<center>💙 HealthAI | Built with Python, Streamlit & Hugging Face</center>",
    unsafe_allow_html=True,
)
 
Claude finished the response
