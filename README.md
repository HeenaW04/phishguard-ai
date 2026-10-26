# PhishGuard AI — Phishing URL Detection Using Machine Learning

This ready-made student project extracts lexical features from a URL string, trains multiple classifiers on a public dataset, compares results, and displays predictions in Streamlit.

**Safety/accuracy limitation:** the app does not visit the URL, inspect HTML, check DNS/certificates or query live reputation. It cannot guarantee safety. It is a classroom prototype only.

## Dataset
UCI PhiUSIIL Phishing URL Dataset (ID 967), with labelled URL/webpage records. Dataset labels: 0 = phishing, 1 = legitimate. Official page: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

## Project files
- `app.py`: Streamlit interface
- `train_model.py`: download dataset, prepare features, train and compare models
- `requirements.txt`: pinned dependencies
- `PhishGuard_AI_Training.ipynb`: Colab notebook to train models
- `REPORT_OUTLINE.md`: report scaffold
- `RESEARCH_REFERENCES.md`: dataset/research sources

## Beginner workflow (Google Colab, no local Python needed)
1. Extract this ZIP on your computer.
2. Open https://colab.research.google.com and upload `PhishGuard_AI_Training.ipynb`.
3. Run notebook cells from top to bottom. When asked, upload this project ZIP.
4. The notebook installs packages, downloads the UCI data, trains Random Forest, Extra Trees, Logistic Regression, SVM and XGBoost, and downloads `trained_models.zip`.
5. Extract `trained_models.zip`. Copy its `models` folder into this project next to `app.py`.
6. Create a GitHub repository named `phishguard-ai` and upload `app.py`, `requirements.txt`, `README.md`, `REPORT_OUTLINE.md`, `RESEARCH_REFERENCES.md`, and the generated `models/` folder.
7. Go to https://share.streamlit.io, create an app, select repository/branch `main`, main file `app.py`, and choose Python 3.12 in advanced settings if available. Deploy.
8. Test a few URLs and scroll down to verify the model comparison table.

## If training locally
Use Python 3.12, then run `pip install -r requirements.txt`, `python train_model.py`, and `streamlit run app.py`.

## Expected output files after training
`models/model_bundle.joblib`, `models/model_comparison.csv`, and `models/training_summary.json`.

## Faculty demo checklist
Problem statement; dataset and label mapping; data cleaning; train/test split; feature extraction; comparison table; prediction UI; metrics; limitations and responsible use. Use actual values from generated files—do not make up results.
