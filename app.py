
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ PhishGuard AI")
st.header("Machine Learning Based Phishing URL Detection")
st.write(
    "Analyze a website URL using trained machine learning "
    "models to identify potentially suspicious addresses."
)

st.warning(
    "Educational prototype only. This tool analyzes URL text; "
    "it does not visit websites, inspect webpage content, or "
    "check live security reputation. Predictions can be wrong."
)


# --------------------------------------------------
# Load trained model bundle
# --------------------------------------------------
MODEL_PATH = Path(__file__).parent / "models" / "model_bundle.joblib"
COMPARISON_PATH = Path(__file__).parent / "models" / "model_comparison.csv"


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {MODEL_PATH}. "
            "Upload model_bundle.joblib into the models folder."
        )

    return joblib.load(MODEL_PATH)


def normalize_url(url):
    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    # Allow users to enter example.com without https://
    if "://" not in url:
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(
            "Enter a valid URL, such as https://example.com"
        )

    return url


def get_url_details(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    try:
        ipaddress.ip_address(hostname)
        uses_ip = True
    except ValueError:
        uses_ip = False

    return {
        "Hostname": hostname,
        "HTTPS used": parsed.scheme == "https",
        "URL length": len(url),
        "Subdomain count": max(0, len(hostname.split(".")) - 2),
        "IP address used as host": uses_ip,
        "Non-standard port specified": parsed.port is not None
        and parsed.port not in (80, 443),
    }


# --------------------------------------------------
# URL input and model selection
# --------------------------------------------------
left, right = st.columns([1.4, 1])

with left:
    st.subheader("Enter a URL")

    url_input = st.text_input(
        "Website address",
        placeholder="https://www.example.com",
    )

    bundle = None
    load_error = None

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        load_error = str(exc)

    if load_error:
        st.error(f"Model loading error: {load_error}")

    if bundle is not None:
        models = bundle.get("models", {})

        if not models:
            st.error(
                "No trained classifiers were found inside the model bundle. "
                "Check the training script and saved bundle format."
            )
            selected_model = None
        else:
            model_names = list(models.keys())
            default_name = bundle.get("best_model_name", model_names[0])

            if default_name not in model_names:
                default_name = model_names[0]

            selected_model = st.selectbox(
                "Choose a machine learning model",
                options=model_names,
                index=model_names.index(default_name),
            )

        analyze = st.button(
            "🔎 Analyze URL",
            type="primary",
            use_container_width=True,
        )
    else:
        selected_model = None
        analyze = st.button(
            "🔎 Analyze URL",
            type="primary",
            use_container_width=True,
            disabled=True,
        )

with right:
    st.subheader("How it works")
    st.markdown(
        """
        1. Read and normalize the submitted URL.
        2. Extract character-pattern features using TF-IDF.
        3. Apply the selected trained classifier.
        4. Display the prediction and available benchmark results.
        """
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if analyze and bundle is not None and selected_model is not None:
    try:
        url = normalize_url(url_input)
        model = bundle["models"][selected_model]

        # The trained pipeline performs its own TF-IDF transformation.
        # Do not convert the URL string to float or create numeric
        # features here.
        prediction = model.predict([url])[0]

        st.divider()
        st.subheader("Analysis Result")

        # Training dataset convention:
        # 0 = phishing, 1 = legitimate.
        if str(prediction) == "0":
            st.error("⚠️ Potentially Phishing URL")
        elif str(prediction) == "1":
            st.success("✅ Likely Legitimate URL")
        else:
            st.info(f"Model prediction: {prediction}")

        # Show probabilities when the classifier supports them.
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([url])[0]
            classes = list(model.classes_)

            probability_map = {
                str(label): float(prob)
                for label, prob in zip(classes, probabilities)
            }

            predicted_probability = probability_map.get(
                str(prediction)
            )

            if predicted_probability is not None:
                st.metric(
                    "Model confidence for predicted class",
                    f"{predicted_probability * 100:.2f}%",
                )

        st.caption(
            "This percentage is the model's predicted class probability, "
            "not a verified probability that the website is malicious."
        )

        st.subheader("URL Details")
        st.dataframe(
            pd.DataFrame(
                list(get_url_details(url).items()),
                columns=["Feature", "Value"],
            ),
            hide_index=True,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Unable to analyze this URL: {exc}")


# --------------------------------------------------
# Model benchmark results
# --------------------------------------------------
st.divider()
st.subheader("Model Performance Comparison")

if COMPARISON_PATH.exists():
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.info(f"Could not read the comparison file: {exc}")
else:
    st.info(
        "Model comparison file is not available yet. "
        "Upload model_comparison.csv into the models folder."
    )

st.caption(
    "PhishGuard AI is an educational machine learning prototype. "
    "Do not use its prediction as the only basis for a security decision."
)
