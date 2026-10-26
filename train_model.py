import os
import json
import zipfile
import warnings
import joblib
import pandas as pd

from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

warnings.filterwarnings("ignore")

# 1. Create output folder
OUT = Path("/content/phishguard_models")
OUT.mkdir(parents=True, exist_ok=True)

# 2. Download dataset
print("Downloading UCI PhiUSIIL dataset...")
dataset = fetch_ucirepo(id=967)

features = dataset.data.features
targets = dataset.data.targets

url_column = next(
    (c for c in features.columns if str(c).strip().lower() == "url"),
    None
)

if url_column is None:
    raise ValueError(
        "URL column not found. Available columns: "
        + str(list(features.columns))
    )

urls = features[url_column].astype(str).str.strip()
labels = pd.to_numeric(targets.iloc[:, 0], errors="coerce")

# 3. Prepare data
df = pd.DataFrame({
    "url": urls,
    "label": labels
}).dropna()

df = df[
    (df["url"] != "") &
    (df["label"].isin([0, 1]))
].drop_duplicates(subset="url").reset_index(drop=True)

# UCI PhiUSIIL labels: 0 = phishing, 1 = legitimate
df["label"] = df["label"].astype(int)

# Limit dataset size to reduce Colab memory usage
if len(df) > 60000:
    per_class = min(30000, int(df["label"].value_counts().min()))
    df = (
        df.groupby("label", group_keys=False)
          .sample(n=per_class, random_state=42)
          .sample(frac=1, random_state=42)
          .reset_index(drop=True)
    )

print("\nDataset size:", len(df))
print("\nLabels (0=phishing, 1=legitimate):")
print(df["label"].value_counts().sort_index())

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df["url"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# 5. Define models
models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Complement Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )),
        ("classifier", ComplementNB(alpha=0.5))
    ])
}

# 6. Train and evaluate
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    result = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision_macro": round(
            precision_score(y_test, predictions,
                            average="macro", zero_division=0), 4
        ),
        "Recall_macro": round(
            recall_score(y_test, predictions,
                         average="macro", zero_division=0), 4
        ),
        "F1_macro": round(
            f1_score(y_test, predictions,
                     average="macro", zero_division=0), 4
        )
    }

    # Confusion matrix order: phishing (0), legitimate (1)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    result["Correct_phishing"] = int(cm[0, 0])
    result["Phishing_missed"] = int(cm[0, 1])
    result["Legitimate_flagged_phishing"] = int(cm[1, 0])
    result["Correct_legitimate"] = int(cm[1, 1])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[
            :, list(model.classes_).index(1)
        ]
        result["ROC_AUC_legitimate"] = round(
            roc_auc_score(y_test, probs), 4
        )

    results.append(result)
    trained_models[name] = model

    print(pd.Series(result).to_string())

# 7. Select model with highest macro F1
metrics = pd.DataFrame(results).sort_values(
    "F1_macro", ascending=False
)
best_name = str(metrics.iloc[0]["Model"])

# 8. Save the model bundle
bundle = {
    "models": trained_models,
    "best_model_name": best_name,
    "label_mapping": {
        "0": "Potentially phishing",
        "1": "Likely legitimate"
    },
    "dataset": "UCI PhiUSIIL",
    "input_type": "raw URL text",
    "limitations": (
        "URL-only screening; does not inspect webpage content "
        "or live website reputation."
    )
}

joblib.dump(bundle, OUT / "model_bundle.joblib", compress=3)
metrics.to_csv(OUT / "model_comparison.csv", index=False)

summary = {
    "best_model": best_name,
    "records_used": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "label_mapping": {
        "0": "phishing",
        "1": "legitimate"
    },
    "results": results
}

(OUT / "training_summary.json").write_text(
    json.dumps(summary, indent=2)
)

# 9. Create ZIP containing the generated model files
zip_path = "/content/trained_models.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for file in OUT.iterdir():
        if file.is_file():
            z.write(file, arcname=file.name)

print("\nTRAINING COMPLETED")
print("Best model:", best_name)
print("Saved files:", list(os.listdir(OUT)))
print("ZIP created at:", zip_path)

# 10. Display benchmark table
display(metrics)

# 11. Download the trained files
from google.colab import files
files.download(zip_path)
