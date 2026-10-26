
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_bundle.joblib"
RESULTS_PATH = BASE_DIR / "models" / "model_comparison.csv"

st.title("🛡️ PhishGuard AI")
st.subheader("Machine Learning-Based Phishing URL Detection")

st.write(
    "Analyze URL text using trained machine learning models. "
    "This is a research and educational tool, not a guarantee of website safety."
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_comparison():
    if RESULTS_PATH.exists():
        return pd.read_csv(RESULTS_PATH)
    return pd.DataFrame()

if not MODEL_PATH.exists():
    st.error("Model bundle not found: models/model_bundle.joblib")
    st.info("Upload the trained model bundle to the GitHub models/ folder.")
    st.stop()

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load model bundle: {exc}")
    st.stop()

models = bundle["models"]
best_model_name = bundle.get("best_model_name", list(models)[0])

st.success(f"Loaded {len(models)} trained models.")

st.caption(f"Best model based on validation metric: {best_model_name}")

model_names = list(models.keys())
default_index = (
    model_names.index(best_model_name)
    if best_model_name in model_names else 0
)

selected_model_name = st.selectbox(
    "Choose a trained model",
    model_names,
    index=default_index
)

url = st.text_input(
    "Enter a URL",
    placeholder="https://www.google.com/"
)

def inspect_url(raw_url):
    candidate = raw_url.strip()

    if not candidate:
        return None, "Please enter a URL."

    if any(ch.isspace() for ch in candidate):
        return None, "The URL must not contain spaces."

    # Add a scheme only to make ordinary domain input parseable.
    if "://" not in candidate:
        candidate = "https://" + candidate

    parsed = urlparse(candidate)

    if parsed.scheme not in ("http", "https"):
        return None, "Only HTTP and HTTPS URLs are supported."

    if not parsed.hostname:
        return None, "Please enter a valid hostname."

    try:
        parsed.hostname.encode("idna")
        _ = parsed.port
    except (UnicodeError, ValueError):
        return None, "The hostname or port is invalid."

    return candidate, None

if st.button("Analyze URL", type="primary"):
    normalized_url, error = inspect_url(url)

    if error:
        st.error(error)
    else:
        model = models[selected_model_name]

        # Pass raw text to the complete TF-IDF + classifier pipeline.
        prediction = int(model.predict([normalized_url])[0])

        st.session_state["last_url"] = normalized_url
        st.session_state["last_model"] = selected_model_name
        st.session_state["last_prediction"] = prediction

if "last_prediction" in st.session_state:
    prediction = st.session_state["last_prediction"]
    normalized_url = st.session_state["last_url"]
    selected = st.session_state["last_model"]

    st.markdown("---")
    st.subheader("Prediction result")

    if prediction == 0:
        st.error("⚠️ Model classification: Potentially phishing")
    else:
        st.success("Model classification: Likely legitimate")

    st.caption(
        "This is a model prediction, not a verified safety assessment. "
        "A legitimate-looking URL can still be dangerous, and a "
        "phishing detector can make mistakes."
    )

    st.write("**Analyzed URL:**", normalized_url)
    st.write("**Selected model:**", selected)

    parsed = urlparse(normalized_url)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hostname", parsed.hostname or "Unknown")
        st.metric("HTTPS scheme", "Yes" if parsed.scheme == "https" else "No")
    with col2:
        st.metric("URL length", len(normalized_url))
        st.metric("Subdomain count", max(0, len(
            (parsed.hostname or "").split(".")
        ) - 2))

    host = parsed.hostname or ""
    indicators = {
        "IP address in hostname": bool(
            re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host)
        ),
        "Contains @ symbol": "@" in normalized_url,
        "Contains URL shortener-like host": any(
            short in host.lower()
            for short in ["bit.ly", "tinyurl.com", "t.co", "goo.gl"]
        ),
        "Contains unusually long URL": len(normalized_url) > 100,
    }

    st.subheader("Basic URL indicators")
    st.dataframe(
        pd.DataFrame(
            [{"Indicator": k, "Detected": "Yes" if v else "No"}
             for k, v in indicators.items()]
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "These are simple heuristics, not proof of phishing. "
        "HTTPS and a short URL do not guarantee safety."
    )

st.markdown("---")
st.subheader("Model performance comparison")

comparison = load_comparison()

if not comparison.empty:
    st.dataframe(
        comparison.round(4),
        use_container_width=True,
        hide_index=True
    )

    metric_options = [
        col for col in [
            "Accuracy", "Precision_macro",
            "Recall_macro", "F1_macro", "ROC_AUC"
        ]
        if col in comparison.columns
    ]

    if metric_options:
        metric = st.selectbox(
            "Metric for comparison chart",
            metric_options,
            index=(
                metric_options.index("F1_macro")
                if "F1_macro" in metric_options else 0
            )
        )

        chart = comparison.set_index("Model")[[metric]]
        st.bar_chart(chart)
else:
    st.warning(
        "Model comparison file not found. "
        "Add models/model_comparison.csv to the repository."
    )

st.markdown("---")
st.caption(
    "PhishGuard AI | Academic ML project | "
    "Predictions may be incorrect; do not rely on this app alone "
    "to decide whether to visit a website."
)
