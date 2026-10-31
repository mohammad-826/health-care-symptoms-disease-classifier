import os

import joblib
import streamlit as st
from huggingface_hub import hf_hub_download

# ----------------------------------
# SETTINGS
# ----------------------------------

APP_TITLE = "Health-Care Symptoms Classifier"
DEVELOPER_NAME = "Muhammad"
GITHUB_URL = "https://github.com/mohammad-826"  # change if your username is different
REPO_ID = "Muh340/healthcare-symptoms-disease-classifier"
MODEL_FILENAME = "healthcare_disease_model.pkl"
HF_TOKEN = os.environ.get("HF_TOKEN") or None  # only needed for a private model

st.set_page_config(page_title=APP_TITLE, page_icon="🏥")


# ----------------------------------
# MODEL
# ----------------------------------

@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Download the model from Hugging Face once and keep it in memory.

    The model is now only a few MB (TF-IDF + Logistic Regression),
    so it fits easily inside Streamlit Cloud's memory limit.
    """
    path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME, token=HF_TOKEN)
    return joblib.load(path)


def clear_text():
    st.session_state["symptoms"] = ""


# ----------------------------------
# PAGE
# ----------------------------------

st.title(f"🏥 {APP_TITLE}")
st.caption(
    f"Developed by **{DEVELOPER_NAME}** · [GitHub]({GITHUB_URL}) · "
    f"[Model on Hugging Face](https://huggingface.co/{REPO_ID})"
)

st.write(
    "Type your symptoms separated by commas, for example "
    "*fever, cough, sore throat*, then click **Analyze**."
)

text = st.text_area("Your symptoms", key="symptoms", placeholder="fever, headache, cough")

col1, col2 = st.columns([3, 1])
analyze = col1.button("Analyze", type="primary")
col2.button("Clear", on_click=clear_text)

if analyze:
    if not text.strip():
        st.error("Please enter your symptoms first.")
    else:
        try:
            model = load_model()

            # Check that the model knows at least one of the words entered
            if model.named_steps["tfidf"].transform([text]).nnz == 0:
                st.warning(
                    "None of these words are known to the model. "
                    "Check the spelling or try common symptoms like 'fever' or 'cough'."
                )
            else:
                probs = model.predict_proba([text])[0]
                top5 = sorted(zip(model.classes_, probs), key=lambda x: x[1], reverse=True)[:5]

                st.success(f"Predicted category: **{top5[0][0]}**")
                st.metric("Model confidence", f"{top5[0][1] * 100:.2f}%")

                st.subheader("Top 5 predictions")
                for label, score in top5:
                    st.write(f"{label}: {score * 100:.2f}%")
                    st.progress(float(score))

                st.warning(
                    "This model scores only about 3-4% accuracy on its test set "
                    "(30 diseases, so random guessing is about 3.3%). Treat results as a demo."
                )
        except Exception as e:
            st.error("Could not load or run the model.")
            st.exception(e)

st.divider()
st.caption(
    "⚠️ Educational project only. This is not medical advice, a diagnosis or a treatment "
    "recommendation. Always consult a qualified healthcare professional."
)
