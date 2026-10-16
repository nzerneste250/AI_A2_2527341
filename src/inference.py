import joblib
import pandas as pd
from src.config import FEATURES,MODELS,VERIFICATION_CODE
def load_bundle():
    paths=[MODELS/"regression.joblib",MODELS/"classification.joblib",MODELS/"clustering.joblib"]
    if not all(p.exists() for p in paths): raise FileNotFoundError("Saved models missing; run run_all.py first")
    return tuple(joblib.load(p) for p in paths)
def predict_record(record):
    missing=[f for f in FEATURES if f not in record]
    if missing: raise ValueError(f"Missing required fields: {missing}")
    X=pd.DataFrame([{f:record[f] for f in FEATURES}]); reg,clf,cluster=load_bundle();
    return {"estimated_harvest_tons":float(reg.predict(X)[0]),"predicted_risk_class":int(clf.predict(X)[0]),"high_risk_probability":float(clf.predict_proba(X)[0,1]),"cluster":int(cluster.predict(X)[0]),"verification_code":VERIFICATION_CODE}
