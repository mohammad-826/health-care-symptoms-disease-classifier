import streamlit as st

from src.model_loader import load_model
from src.predictor import predict_disease


st.set_page_config(
    page_title="Healthcare Classifier",
    page_icon="🏥",
    layout="centered"
)

st.title("Healthcare Symptoms Classifier")

st.write(
    "Enter a text description of symptoms "
    "to see the model's predicted category."
)

st.warning(
    "Educational demonstration only. "
    "This tool cannot diagnose diseases or replace "
    "a qualified healthcare professional."
)

symptoms = st.text_area(
    "Describe the symptoms:",
    placeholder="Enter a text description..."
)

if st.button("Predict"):

    if symptoms.strip():

        with st.spinner("Loading model and processing text..."):

            tokenizer, model = load_model()

            disease, confidence = predict_disease(
                symptoms,
                tokenizer,
                model
            )

        st.subheader("Model Prediction")

        st.write("Predicted category:", disease)

        st.write(
            f"Model confidence: {confidence * 100:.2f}%"
        )

        st.info(
            "This is an experimental model output, "
            "not a medical diagnosis."
        )

    else:
        st.error("Please enter a description first.")
