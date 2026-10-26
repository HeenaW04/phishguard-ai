# PhishGuard AI report outline

## Abstract
This project builds a machine-learning prototype that screens URL strings using lexical URL characteristics. It compares several classifiers and presents predictions through a Streamlit interface. It is an educational tool, not a guarantee of website safety.

## 1. Introduction
Explain phishing, deceptive links, and the value and limitations of URL screening.

## 2. Problem statement
Explore whether URL-text patterns can help distinguish phishing and legitimate URLs.

## 3. Objectives
- Clean and deduplicate labelled URL data.
- Extract URL lexical features.
- Compare Logistic Regression, SVM, Random Forest, Extra Trees, and XGBoost.
- Evaluate accuracy, macro precision/recall/F1 and ROC-AUC.
- Deploy a Streamlit interface.

## 4. Dataset
UCI PhiUSIIL Phishing URL Dataset, ID 967: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset. Label 0 = phishing, Label 1 = legitimate. Use the actual counts from `models/training_summary.json`.

## 5. Methodology
Describe deduplication, URL-text feature extraction, stratified 80/20 train/test split, model training, metrics and best-model selection. The prototype does not inspect page HTML, DNS or live reputation.

## 6. System design
URL input → feature extraction → trained classifier → prediction/score → benchmark comparison.

## 7. Implementation
Python, pandas, scikit-learn, XGBoost, joblib and Streamlit. Add screenshots.

## 8. Results
Insert the actual `models/model_comparison.csv` table. Do not invent metrics.

## 9. Limitations
Lexical features can cause false positives and false negatives; no live page inspection; model scores are not calibrated guarantees. Do not use as the only security control.

## 10. Conclusion and future work
Summarize measured results. Possible future extensions include safe reputation feeds, sandboxed page-level analysis and explainability.

## References
See `RESEARCH_REFERENCES.md`.
