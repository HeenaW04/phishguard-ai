from pathlib import Path
import json, warnings
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent; OUT=ROOT/"models"; OUT.mkdir(exist_ok=True)
MAX_SAMPLES=60000; SEED=42

def features(raw):
    from urllib.parse import urlparse
    import ipaddress
    value=str(raw).strip()
    if "://" not in value: value="https://"+value
    p=urlparse(value); host=(p.hostname or "").lower(); path=p.path or ""; query=p.query or ""
    try: ipaddress.ip_address(host); ip=1
    except ValueError: ip=0
    labels=[x for x in host.split(".") if x]; full=value.lower()
    terms=["login","verify","secure","account","update","password","confirm","wallet","banking","signin","webscr"]
    short={"bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly","cutt.ly","rebrand.ly"}
    return {"url_length":len(value),"hostname_length":len(host),"path_length":len(path),"query_length":len(query),
    "count_dots":value.count("."),"count_hyphens":value.count("-"),"count_at":value.count("@"),
    "count_question":value.count("?"),"count_equal":value.count("="),"count_percent":value.count("%"),
    "count_digits":sum(c.isdigit() for c in value),"count_slashes":value.count("/"),
    "count_subdomains":max(0,len(labels)-2),"https_scheme":int(p.scheme.lower()=="https"),
    "has_ip_address":ip,"suspicious_keyword_count":sum(t in full for t in terms),
    "uses_url_shortener":int(host in short),"has_double_slash_in_path":int("//" in path)}

def main():
    from ucimlrepo import fetch_ucirepo
    print("Downloading UCI PhiUSIIL dataset (internet required)...")
    ds=fetch_ucirepo(id=967); raw=ds.data.features.copy(); target=ds.data.targets.copy()
    urlcol=next((c for c in raw.columns if str(c).strip().lower()=="url"),None)
    if urlcol is None: raise ValueError("URL column not found: "+str(list(raw.columns)))
    urls=raw[urlcol].astype(str).reset_index(drop=True)
    y=pd.to_numeric(target.iloc[:,0],errors="coerce").reset_index(drop=True)
    df=pd.DataFrame({"URL":urls,"target":y}).dropna().drop_duplicates("URL")
    df=df[df.target.isin([0,1])].reset_index(drop=True)
    if len(df)>MAX_SAMPLES:
        per=min(MAX_SAMPLES//2,int(df.target.value_counts().min()))
        df=df.groupby("target",group_keys=False).sample(n=per,random_state=SEED).reset_index(drop=True)
    print("Unique records used:",len(df))
    X=pd.DataFrame([features(u) for u in df.URL]); y=df.target.astype(int); cols=list(X.columns)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    models={
      "Random Forest":RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=SEED,n_jobs=-1),
      "Extra Trees":ExtraTreesClassifier(n_estimators=200,class_weight="balanced",random_state=SEED,n_jobs=-1),
      "Logistic Regression":Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=1200,class_weight="balanced",random_state=SEED))]),
      "SVM (RBF)":Pipeline([("scale",StandardScaler()),("model",SVC(C=2,probability=True,class_weight="balanced",random_state=SEED))])}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"]=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=SEED,n_jobs=-1)
    except Exception as e: print("XGBoost skipped:",e)
    rows=[]; fitted={}
    for name,m in models.items():
        print("Training",name); m.fit(Xtr,ytr); pred=m.predict(Xte)
        row={"Model":name,"Accuracy":accuracy_score(yte,pred),"Precision_macro":precision_score(yte,pred,average="macro",zero_division=0),"Recall_macro":recall_score(yte,pred,average="macro",zero_division=0),"F1_macro":f1_score(yte,pred,average="macro",zero_division=0)}
        if hasattr(m,"predict_proba"):
            pr=m.predict_proba(Xte)[:,list(m.classes_).index(1)]; row["ROC_AUC_legitimate"]=roc_auc_score(yte,pr)
        rows.append(row); fitted[name]=m; print(row)
    metrics=pd.DataFrame(rows).sort_values("F1_macro",ascending=False)
    best=str(metrics.iloc[0]["Model"])
    joblib.dump({"models":fitted,"best_model_name":best,"feature_columns":cols,"label_mapping":{"0":"Potentially phishing","1":"Likely legitimate"}},OUT/"model_bundle.joblib",compress=3)
    metrics.to_csv(OUT/"model_comparison.csv",index=False)
    (OUT/"training_summary.json").write_text(json.dumps({"dataset":"UCI PhiUSIIL Phishing URL Dataset","uci_id":967,"unique_records_used":len(df),"train_records":len(Xtr),"test_records":len(Xte),"features":cols,"best_model":best,"labels":{"0":"phishing","1":"legitimate"},"limitation":"URL-text features only; no live webpage inspection."},indent=2))
    print("\\nSaved model bundle and benchmark to",OUT)
if __name__=="__main__": main()
