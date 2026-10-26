
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================================================
# PHISHGUARD AI — STREAMLIT APPLICATION
# Four models: Logistic Regression, SVM, Decision Tree,
# Random Forest
# =========================================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

BUNDLE_PATH = MODELS_DIR / "model_bundle.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
SUMMARY_PATH = MODELS_DIR / "training_summary.json"

EXPECTED_MODELS = [
    "Logistic Regression",
    "SVM",
    "Decision Tree",
    "Random Forest"
]

# ---------------- LOAD ARTIFACTS ----------------

@st.cache_resource
def load_bundle(path, modified_time):
    return joblib.load(path)

@st.cache_data
def load_csv(path, modified_time):
    return pd.read_csv(path)

@st.cache_data
def load_json(path, modified_time):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_file_mtime(path):
    return path.stat().st_mtime if path.exists() else 0


st.markdown(
    """
    <div style="padding: 8px 0 12px 0;">
      <h1 style="margin-bottom:0;">🛡️ PhishGuard AI</h1>
      <h3 style="font-weight:400; margin-top:5px;">
        Machine Learning-Based Phishing URL Detection
      </h3>
      <p>
        Analyze URL patterns using trained machine learning models,
        compare their performance, and explore the training results.
      </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Academic project | Python · Scikit-learn · TF-IDF · Streamlit"
)

if not BUNDLE_PATH.exists():
    st.error(
        "Model bundle not found. Expected file: "
        "models/model_bundle.joblib"
    )
    st.stop()

try:
    bundle = load_bundle(
        str(BUNDLE_PATH), get_file_mtime(BUNDLE_PATH)
    )
    models = bundle["models"]
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

if not isinstance(models, dict) or not models:
    st.error("The model bundle does not contain trained models.")
    st.stop()

# Load comparison and summary files if available.
comparison = pd.DataFrame()
summary = {}

if COMPARISON_PATH.exists():
    try:
        comparison = load_csv(
            str(COMPARISON_PATH), get_file_mtime(COMPARISON_PATH)
        )
    except Exception as exc:
        st.warning(f"Could not read model comparison CSV: {exc}")

if SUMMARY_PATH.exists():
    try:
        summary = load_json(
            str(SUMMARY_PATH), get_file_mtime(SUMMARY_PATH)
        )
    except Exception as exc:
        st.warning(f"Could not read training summary: {exc}")

model_names = list(models.keys())
best_model_name = bundle.get("best_model_name", model_names[0])

st.success(f"Loaded {len(model_names)} trained model(s).")

if set(EXPECTED_MODELS).issubset(set(model_names)):
    st.success("All four required models are available.")
else:
    missing = sorted(set(EXPECTED_MODELS) - set(model_names))
    st.warning(
        "Some expected models are missing from the bundle: "
        + ", ".join(missing)
    )

# ---------------- HELPER FUNCTIONS ----------------

def validate_url(raw_url):
    candidate = raw_url.strip()

    if not candidate:
        return None, "Please enter a URL."

    if any(ch.isspace() for ch in candidate):
        return None, "The URL must not contain spaces."

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


def url_indicators(url):
    parsed = urlparse(url)
    host = parsed.hostname or ""

    try:
        import ipaddress
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False

    shorteners = [
        "bit.ly", "tinyurl.com", "t.co", "goo.gl",
        "is.gd", "ow.ly", "buff.ly"
    ]

    return {
        "IP address used as hostname": is_ip,
        "Contains @ symbol": "@" in url,
        "Shortener-like hostname": any(
            host.lower() == domain or host.lower().endswith("." + domain)
            for domain in shorteners
        ),
        "URL longer than 100 characters": len(url) > 100,
        "Contains multiple subdomain levels": len(host.split(".")) > 3,
        "Contains an explicit port": parsed.port is not None,
        "Uses HTTPS": parsed.scheme == "https",
    }


def format_metric(value):
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "N/A"


def get_model_row(name):
    if comparison.empty or "Model" not in comparison.columns:
        return None

    rows = comparison[comparison["Model"] == name]
    if rows.empty:
        return None
    return rows.iloc[0]


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Project Information")
    st.write("**Project:** PhishGuard AI")
    st.write("**Task:** Binary URL classification")
    st.write("**Features:** Character-level TF-IDF")
    st.write("**Dataset:** UCI PhiUSIIL")
    st.write("**Models available:**", len(model_names))

    st.divider()
    st.caption(
        "A model prediction is not a verified security assessment. "
        "Do not visit suspicious links just to test them."
    )


# ---------------- FOUR MAIN TABS ----------------

tab_detect, tab_compare, tab_eda, tab_about = st.tabs([
    "🛡️ URL Detection",
    "📊 Model Comparison",
    "🔎 Dataset & EDA",
    "ℹ️ About Project"
])


# =========================================================
# TAB 1 — URL DETECTION
# =========================================================

with tab_detect:
    st.header("Phishing URL Detector")
    st.write(
        "Choose one of the trained models and enter a URL. "
        "The selected pipeline processes the raw URL string."
    )

    default_index = (
        model_names.index(best_model_name)
        if best_model_name in model_names else 0
    )

    selected_model = st.selectbox(
        "Select a trained model",
        model_names,
        index=default_index,
        key="detection_model"
    )

    url_input = st.text_input(
        "Enter a URL",
        placeholder="https://www.google.com/",
        key="url_input"
    )

    analyze = st.button(
        "🔍 Analyze URL",
        type="primary",
        key="analyze_url"
    )

    if analyze:
        normalized_url, error = validate_url(url_input)

        if error:
            st.error(error)
        else:
            try:
                prediction = int(
                    models[selected_model].predict([normalized_url])[0]
                )

                st.session_state["prediction_result"] = {
                    "url": normalized_url,
                    "model": selected_model,
                    "prediction": prediction
                }

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("prediction_result")

    if result:
        st.divider()
        st.subheader("Prediction Result")

        pred = result["prediction"]
        result_url = result["url"]
        result_model = result["model"]

        if pred == 0:
            st.error("⚠️ Model classification: Potentially phishing")
        elif pred == 1:
            st.success("Model classification: Likely legitimate")
        else:
            st.warning(f"Model returned an unexpected label: {pred}")

        st.write("**Analyzed URL:**", result_url)
        st.write("**Model used:**", result_model)

        st.info(
            "This result is a machine-learning prediction, not proof "
            "that a URL is safe or malicious. False positives and "
            "false negatives are possible."
        )

        parsed = urlparse(result_url)
        indicators = url_indicators(result_url)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("URL Length", len(result_url))
        c2.metric("HTTPS", "Yes" if parsed.scheme == "https" else "No")
        c3.metric("Hostname", parsed.hostname or "Unknown")
        c4.metric(
            "Subdomain Levels",
            max(0, len((parsed.hostname or "").split(".")) - 2)
        )

        st.subheader("Basic URL Indicators")
        indicator_df = pd.DataFrame([
            {
                "Indicator": key,
                "Detected": "Yes" if value else "No"
            }
            for key, value in indicators.items()
        ])

        st.dataframe(
            indicator_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Indicators are simple URL heuristics. They are not the "
            "features used as a separate classifier, and they do not "
            "prove that a URL is malicious or safe."
        )

        row = get_model_row(result_model)

        if row is not None:
            st.subheader("Selected Model's Test Metrics")

            metric_columns = [
                ("Accuracy", "Accuracy"),
                ("Precision", "Precision_macro"),
                ("Recall", "Recall_macro"),
                ("F1-score", "F1_macro"),
                ("ROC-AUC", "ROC_AUC")
            ]

            cols = st.columns(len(metric_columns))

            for col, (label, column) in zip(cols, metric_columns):
                if column in row.index:
                    col.metric(label, format_metric(row[column]))
                else:
                    col.metric(label, "N/A")


# =========================================================
# TAB 2 — MODEL COMPARISON
# =========================================================

with tab_compare:
    st.header("Model Performance Dashboard")
    st.write(
        "Compare the four trained classifiers using the metrics "
        "calculated on the held-out test set."
    )

    if comparison.empty:
        st.warning(
            "Model comparison CSV is missing or empty. Upload "
            "models/model_comparison.csv to GitHub."
        )
    elif "Model" not in comparison.columns:
        st.error("The CSV must contain a 'Model' column.")
    else:
        # Metric cards for each model.
        st.subheader("Model Summary")

        for start in range(0, len(comparison), 2):
            cols = st.columns(2)

            for offset, col in enumerate(cols):
                idx = start + offset
                if idx >= len(comparison):
                    continue

                row = comparison.iloc[idx]
                name = str(row["Model"])

                with col:
                    st.markdown(f"### {name}")
                    m1, m2 = st.columns(2)

                    if "Accuracy" in row.index:
                        m1.metric(
                            "Accuracy", format_metric(row["Accuracy"])
                        )

                    if "F1_macro" in row.index:
                        m2.metric(
                            "Macro F1", format_metric(row["F1_macro"])
                        )

                    if "Precision_macro" in row.index:
                        st.write(
                            "Macro precision:",
                            format_metric(row["Precision_macro"])
                        )

                    if "Recall_macro" in row.index:
                        st.write(
                            "Macro recall:",
                            format_metric(row["Recall_macro"])
                        )

                    if "ROC_AUC" in row.index:
                        st.write(
                            "ROC-AUC:", format_metric(row["ROC_AUC"])
                        )

                    if name == best_model_name:
                        st.caption("Best model recorded in bundle metadata")

                    st.divider()

        st.subheader("Full Comparison Table")

        display_df = comparison.copy()

        numeric_cols = display_df.select_dtypes(
            include="number"
        ).columns

        st.dataframe(
            display_df.style.format(
                {col: "{:.4f}" for col in numeric_cols}
            ),
            use_container_width=True,
            hide_index=True
        )

        chart_metrics = [
            col for col in [
                "Accuracy", "Precision_macro", "Recall_macro",
                "F1_macro", "ROC_AUC"
            ]
            if col in comparison.columns
        ]

        if chart_metrics:
            st.subheader("Interactive Metric Comparison")

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                chart_metrics,
                index=(
                    chart_metrics.index("F1_macro")
                    if "F1_macro" in chart_metrics else 0
                ),
                key="comparison_metric"
            )

            chart_df = comparison[["Model", selected_metric]].copy()
            chart_df = chart_df.set_index("Model")

            st.bar_chart(chart_df)

            st.caption(
                "Higher scores are generally better, but model choice "
                "should consider false positives, false negatives, "
                "class balance, and performance on unseen URLs."
            )

        # Confusion matrices saved in training_summary.json.
        st.subheader("Confusion Matrices")

        cms = summary.get("confusion_matrices", {})

        if cms:
            cm_model = st.selectbox(
                "Select model for confusion matrix",
                [name for name in model_names if name in cms],
                key="cm_model"
            )

            cm = cms[cm_model]

            if len(cm) == 2 and all(len(row) == 2 for row in cm):
                fig, ax = plt.subplots(figsize=(5, 4))
                image = ax.imshow(cm, interpolation="nearest")

                ax.set_title(f"{cm_model} — Confusion Matrix")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("Actual label")
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels([
                    "Phishing (0)", "Legitimate (1)"
                ])
                ax.set_yticklabels([
                    "Phishing (0)", "Legitimate (1)"
                ])

                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j, i, str(cm[i][j]),
                            ha="center", va="center"
                        )

                fig.colorbar(image, ax=ax)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.caption(
                    "Rows are actual classes and columns are predicted "
                    "classes. Confirm the label order against the "
                    "training script."
                )
            else:
                st.warning("Confusion matrix has an unexpected shape.")
        else:
            st.info(
                "Confusion matrices are not available in training_summary.json. "
                "Upload the summary file generated by your training script "
                "to display them."
            )

        st.caption(
            "Metrics shown here are from the saved experiment results. "
            "They are not live evaluations of the URL entered in Tab 1."
        )


# =========================================================
# TAB 3 — DATASET AND EDA
# =========================================================

with tab_eda:
    st.header("Dataset & Exploratory Data Analysis")

    st.write(
        "This section summarizes the available training metadata. "
        "The charts below are shown only when the corresponding "
        "information was saved during training."
    )

    n_total = summary.get("samples_used")
    n_train = summary.get("train_samples")
    n_test = summary.get("test_samples")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records Used",
        f"{n_total:,}" if isinstance(n_total, (int, float)) else "N/A"
    )
    c2.metric(
        "Training Records",
        f"{n_train:,}" if isinstance(n_train, (int, float)) else "N/A"
    )
    c3.metric(
        "Testing Records",
        f"{n_test:,}" if isinstance(n_test, (int, float)) else "N/A"
    )

    st.subheader("Preprocessing Summary")

    st.markdown(
        """
        - Removed missing or empty URL entries.
        - Removed duplicate URLs.
        - Retained valid binary target labels.
        - Used a stratified train/test split.
        - Used character-level TF-IDF n-grams.
        - Fitted TF-IDF within each model pipeline using training data.
        """
    )

    st.subheader("Feature Extraction")

    st.write(
        "Character TF-IDF represents URLs using weighted character "
        "sequences. These features can capture URL patterns without "
        "requiring a separate manual feature vector."
    )

    feature_info = bundle.get("feature_method", {})
    if isinstance(feature_info, dict):
        st.json(feature_info)
    elif feature_info:
        st.write(feature_info)

    # Class distribution is displayed only if explicitly saved.
    class_counts = summary.get("class_distribution")

    st.subheader("Target Class Distribution")

    if isinstance(class_counts, dict) and class_counts:
        counts_df = pd.DataFrame({
            "Class": list(class_counts.keys()),
            "Count": list(class_counts.values())
        }).set_index("Class")

        st.bar_chart(counts_df)
    else:
        st.info(
            "Class counts were not saved in the current training summary. "
            "For a true dataset-distribution chart, add class counts "
            "during training and regenerate training_summary.json."
        )

    st.subheader("URL Length Analysis")

    length_stats = summary.get("url_length_stats")

    if isinstance(length_stats, dict) and length_stats:
        st.json(length_stats)
    else:
        st.info(
            "URL length statistics were not saved in the training "
            "summary. The detector shows length for an individual "
            "entered URL, but that is not a dataset-level EDA result."
        )

    st.subheader("Dataset Source")
    st.markdown(
        "[UCI PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)"
    )
    st.write("Dataset ID: 967")


# =========================================================
# TAB 4 — ABOUT PROJECT
# =========================================================

with tab_about:
    st.header("About PhishGuard AI")

    st.subheader("Problem Statement")

    st.write(
        "Phishing URLs can imitate legitimate services to deceive "
        "users. PhishGuard AI applies supervised machine learning "
        "to classify URL text as potentially phishing or likely "
        "legitimate."
    )

    st.subheader("Project Objectives")

    st.markdown(
        """
        1. Clean and preprocess a public URL dataset.
        2. Represent URL strings using character-level TF-IDF.
        3. Train and compare four supervised ML models.
        4. Evaluate performance using classification metrics.
        5. Deploy an interactive application using Streamlit.
        """
    )

    st.subheader("Algorithms")

    st.markdown(
        """
        - **Logistic Regression:** A linear classification model.
        - **Support Vector Machine:** Finds a decision boundary
          with a large margin.
        - **Decision Tree:** Learns a sequence of feature-based splits.
        - **Random Forest:** Combines multiple decision trees.
        """
    )

    st.subheader("Technology Stack")

    st.markdown(
        """
        - Python
        - Pandas and NumPy
        - Scikit-learn
        - TF-IDF character n-grams
        - Streamlit
        - Google Colab
        - GitHub
        - Streamlit Community Cloud
        """
    )

    st.subheader("Limitations")

    st.markdown(
        """
        - The model analyzes URL text; it does not inspect the live webpage.
        - Predictions can contain false positives and false negatives.
        - Performance may change on URLs that differ from the training data.
        - HTTPS does not guarantee that a website is trustworthy.
        - Model evaluation scores are not a guarantee of real-world safety.
        """
    )

    st.subheader("Model Metadata")

    st.write("Dataset:", bundle.get("dataset", "Not recorded"))
    st.write("UCI dataset ID:", bundle.get("uci_dataset_id", "Not recorded"))
    st.write("Input format:", bundle.get("input_type", "Not recorded"))
    st.write("Best model recorded:", best_model_name)

    st.divider()

    st.caption(
        "PhishGuard AI is an academic demonstration project. "
        "Do not rely on its output as the sole basis for security decisions."
    )
