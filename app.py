
import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
METRICS_PATH = ROOT / "models" / "model_comparison.csv"

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)


def normalize_url(raw_url):
    url = str(raw_url).strip().lower()

    if not url:
        raise ValueError("Please enter a website URL.")

    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)

    if not parsed.hostname:
        raise ValueError(
            "Invalid URL. Try entering example.com"
        )

    try:
        ipaddress.ip_address(parsed.hostname)
        host_type = "IP address"
    except ValueError:
        host_type = "Domain name"

    return url, parsed, host_type


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


st.title("🛡️ PhishGuard AI")
st.subheader("Machine Learning Based Phishing URL Detection")

st.write(
    "Analyze URL text using machine learning to identify "
    "potentially suspicious website addresses."
)

st.warning(
    "Educational prototype only. This tool analyzes URL text; "
    "it does not visit websites, inspect webpage content, or "
    "check live security reputation. Predictions can be wrong."
)

bundle = load_model()

if bundle is None:
    st.error(
        "Trained model not found. Please add "
        "models/model_bundle.joblib to your GitHub repository."
    )
    st.stop()

models = bundle.get("models", {})

if not models:
    st.error("No trained classifiers were found in the model file.")
    st.stop()

model_names = list(models.keys())
best_name = bundle.get("best_model_name", model_names[0])

if best_name not in model_names:
    best_name = model_names[0]

left, right = st.columns(2)

with left:
    st.markdown("### Enter a URL")

    url_input = st.text_input(
        "Website address",
        placeholder="https://www.example.com"
    )

    selected_model = st.selectbox(
        "Choose a machine learning model",
        model_names,
        index=model_names.index(best_name)
    )

    analyze = st.button(
        "🔍 Analyze URL",
        type="primary",
        use_container_width=True
    )

with right:
    st.markdown("### How it works")

    st.markdown(
        """
        1. Read and normalize the submitted URL.
        2. Extract character-pattern features using TF-IDF.
        3. Apply the selected trained classifier.
        4. Display the prediction and model benchmark results.
        """
    )

if analyze:
    try:
        normalized_url, parsed, host_type = normalize_url(
            url_input
        )

        classifier = models[selected_model]
        prediction = int(classifier.predict([normalized_url])[0])

        st.markdown("---")
        st.markdown("### Analysis Result")

        st.write("**Hostname:**", parsed.hostname)
        st.write("**Host type:**", host_type)

        if prediction == 0:
            st.error("⚠️ Prediction: Potentially phishing")
            st.write(
                "Avoid entering passwords, payment information, "
                "or verification codes until the website is "
                "independently verified."
            )
        elif prediction == 1:
            st.success("✅ Prediction: Likely legitimate")
            st.write(
                "This prediction does not guarantee that the "
                "website is safe."
            )
        else:
            st.warning("The model returned an unexpected label.")

        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(
                [normalized_url]
            )[0]

            class_index = list(classifier.classes_).index(
                prediction
            )

            score = float(probabilities[class_index])

            st.metric(
                "Model score for predicted class",
                f"{score:.1%}"
            )

            st.caption(
                "This is a model probability estimate, not a "
                "verified website safety score."
            )

        with st.expander("View URL details"):
            st.write({
                "Normalized URL": normalized_url,
                "Hostname": parsed.hostname,
                "Protocol": parsed.scheme,
                "Path": parsed.path
            })

    except Exception as error:
        st.error(f"Unable to analyze this URL: {error}")

st.divider()
st.markdown("### Model Performance Comparison")

if METRICS_PATH.exists():
    try:
        metrics = pd.read_csv(METRICS_PATH)

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        if "F1_macro" in metrics.columns:
            st.markdown("#### Macro F1-score")
            st.bar_chart(
                metrics.set_index("Model")["F1_macro"]
            )

    except Exception as error:
        st.warning(f"Could not load benchmark results: {error}")
else:
    st.info(
        "Model comparison will appear after the training script "
        "generates model_comparison.csv."
    )

st.caption(
    "PhishGuard AI is a machine-learning demonstration. "
    "URL-only detection can produce false positives and "
    "false negatives. Do not rely on it as your only security check."
)
