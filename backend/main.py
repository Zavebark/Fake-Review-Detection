from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE=Path(__file__).resolve().parent
MODEL_PATH=BASE/"artifacts"/"review_model.joblib"
app=FastAPI(title="Deceptive Review Detection API", version="1.0.0",
            description="Returns a calibrated-style human-likelihood score and explanation cues.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

bundle=None
def get_bundle():
    global bundle
    if bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503, detail="Model not trained. Run backend/train.py first.")
        bundle=joblib.load(MODEL_PATH)
    return bundle

class ReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    rating: float | None = None
    reviewer_id: str | None = None
    product_id: str | None = None
    timestamp: str | None = None

class BatchRequest(BaseModel):
    reviews: list[ReviewRequest]

def linguistic_features(text):
    words=re.findall(r"\b\w+\b",text.lower())
    sentences=[s for s in re.split(r"[.!?]+",text) if s.strip()]
    return {
      "char_count":len(text),"word_count":len(words),
      "avg_word_length":float(np.mean([len(w) for w in words])) if words else 0.0,
      "sentence_count":len(sentences),"exclamation_count":text.count("!"),
      "question_count":text.count("?"),
      "uppercase_ratio":sum(c.isupper() for c in text)/max(1,sum(c.isalpha() for c in text)),
      "punctuation_ratio":sum(not c.isalnum() and not c.isspace() for c in text)/max(1,len(text)),
      "unique_word_ratio":len(set(words))/max(1,len(words))
    }

def explain(text, score):
    # These are observable cues, not causal explanations from the model.
    cues=[]
    words=re.findall(r"\b\w+\b",text.lower())
    if len(words)<8: cues.append("very short text")
    if text.count("!")>=3: cues.append("frequent exclamation marks")
    if sum(c.isupper() for c in text)/max(1,sum(c.isalpha() for c in text))>0.25: cues.append("high uppercase ratio")
    if words and len(set(words))/len(words)<0.45: cues.append("repetitive vocabulary")
    if not cues: cues.append("overall language pattern")
    return cues

def make_row(req, numeric_columns, config):
    text=req.text
    f=linguistic_features(text)
    f["rating"]=req.rating if req.rating is not None else np.nan
    # At inference, per-reviewer/product counts are unknown unless the extension
    # sends aggregate metadata. Use missing values; imputer handles them.
    f["reviewer_review_count"]=np.nan
    f["product_review_count"]=np.nan
    ts=pd.to_datetime(req.timestamp,errors="coerce",utc=True) if req.timestamp else pd.NaT
    f["timestamp_hour"]=ts.hour if not pd.isna(ts) else np.nan
    f["timestamp_dayofweek"]=ts.dayofweek if not pd.isna(ts) else np.nan
    f["timestamp_epoch_days"]=ts.value/86400000000000 if not pd.isna(ts) else np.nan
    f["hours_since_previous_reviewer_review"]=np.nan
    return pd.DataFrame([{c:f.get(c,np.nan) for c in numeric_columns}])

def predict_one(req):
    b=get_bundle()
    row=make_row(req,b["numeric_columns"],b["config"])
    xt=b["vectorizer"].transform([req.text])
    xn=b["imputer"].transform(row)
    xn=b["scaler"].transform(xn)
    from scipy.sparse import hstack, csr_matrix
    X=hstack([xt,csr_matrix(xn)]).tocsr()
    human_prob=float(b["model"].predict_proba(X)[0,1])
    score=round(10*human_prob,1)
    if score>=8.5: label="Most Likely Human"
    elif score>=7.0: label="Probably Human"
    elif score>=5.0: label="Uncertain"
    elif score>=3.0: label="Probably Fake"
    else: label="Most Likely Fake"
    return {"score":score,"label":label,"human_probability":round(human_prob,4),
            "explanation_cues":explain(req.text,score),
            "notice":"Model estimate, not proof of authenticity."}

@app.get("/health")
def health():
    return {"status":"ok","model_trained":MODEL_PATH.exists()}

@app.post("/predict")
def predict(req: ReviewRequest):
    if not req.text.strip(): raise HTTPException(status_code=422,detail="Review text cannot be empty.")
    return predict_one(req)

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if len(req.reviews)>200: raise HTTPException(status_code=413,detail="Maximum 200 reviews per batch.")
    return {"results":[predict_one(r) for r in req.reviews]}
