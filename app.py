import joblib, pandas as pd, streamlit as st
from pathlib import Path
from urllib.parse import urlparse
import ipaddress

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/"models"/"model_bundle.joblib"
METRICS=ROOT/"models"/"model_comparison.csv"
st.set_page_config(page_title="PhishGuard AI", page_icon="🛡️", layout="wide")

def extract_url_features(raw):
    value=raw.strip()
    if not value: raise ValueError("Enter a URL first.")
    if "://" not in value: value="https://"+value
    p=urlparse(value); host=(p.hostname or "").lower()
    if not host: raise ValueError("Enter a valid URL, such as example.com.")
    try: ipaddress.ip_address(host); has_ip=1
    except ValueError: has_ip=0
    labels=[x for x in host.split(".") if x]
    terms=["login","verify","secure","account","update","password","confirm","wallet","banking","signin","webscr"]
    shorteners={"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly","cutt.ly","rebrand.ly"}
    full=value.lower(); path=p.path or ""; query=p.query or ""
    return {
      "url_length":len(value),"hostname_length":len(host),"path_length":len(path),"query_length":len(query),
      "count_dots":value.count("."),"count_hyphens":value.count("-"),"count_at":value.count("@"),
      "count_question":value.count("?"),"count_equal":value.count("="),"count_percent":value.count("%"),
      "count_digits":sum(c.isdigit() for c in value),"count_slashes":value.count("/"),
      "count_subdomains":max(0,len(labels)-2),"https_scheme":int(p.scheme.lower()=="https"),
      "has_ip_address":has_ip,"suspicious_keyword_count":sum(t in full for t in terms),
      "uses_url_shortener":int(host in shorteners),"has_double_slash_in_path":int("//" in path)
    }

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL) if MODEL.exists() else None

st.title("🛡️ PhishGuard AI")
st.caption("Machine-learning-assisted phishing URL screening")
st.warning("Educational prototype only. This app analyzes URL text; it does not visit the website, inspect HTML, check live reputation, or guarantee safety.")
bundle=load_bundle()
if bundle is None:
    st.error("Trained model files are missing. Follow the Google Colab steps in README.md, then add the generated models/ folder to this project.")
    st.stop()

left,right=st.columns([1,1.1])
with left:
    st.subheader("Check a URL")
    url=st.text_input("Website URL",placeholder="https://example.com/account")
    names=list(bundle["models"])
    selected=st.selectbox("Classifier",names,index=names.index(bundle.get("best_model_name",names[0])))
    run=st.button("Analyze URL",type="primary",use_container_width=True)
    st.caption("The app does not open the submitted URL.")
with right:
    st.subheader("Method")
    st.markdown("1. Extract URL-text features.\\n2. Apply the trained classifier.\\n3. Display prediction and model score.\\n4. Compare models on held-out test data.")
if run:
    try:
        row=pd.DataFrame([extract_url_features(url)],columns=bundle["feature_columns"])
        model=bundle["models"][selected]; pred=int(model.predict(row)[0])
        score=None
        if hasattr(model,"predict_proba"):
            probs=model.predict_proba(row)[0]
            score=float(probs[list(model.classes_).index(pred)])
        if pred==0:
            st.error("⚠️ Prediction: Potentially phishing")
            st.write("Treat this URL cautiously. Do not enter passwords, payment details, or one-time codes.")
        else:
            st.success("✅ Prediction: Likely legitimate")
            st.write("This is not a guarantee that the website is safe.")
        a,b=st.columns(2); a.metric("Classifier",selected); b.metric("Score for predicted class",f"{score:.1%}" if score is not None else "Unavailable")
        with st.expander("Show extracted features"):
            st.dataframe(pd.DataFrame([extract_url_features(url)]).T.rename(columns={0:"Value"}))
    except Exception as e: st.error(f"Could not analyze input: {e}")
st.divider(); st.subheader("Model benchmark")
if METRICS.exists():
    metrics=pd.read_csv(METRICS); st.dataframe(metrics,use_container_width=True,hide_index=True)
    if "F1_macro" in metrics.columns: st.bar_chart(metrics.set_index("Model")["F1_macro"])
else: st.info("Benchmark table appears after training.")
st.caption("Lexical URL features can produce false positives and false negatives. Do not use this prototype as the only security control.")
