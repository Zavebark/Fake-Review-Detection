"""
Train an explainable deceptive-review classifier.
Usage:
  python train.py --csv /path/to/reviews.csv --text-col review_text --label-col label --human-label genuine
Optional:
  --reviewer-col reviewer_id --product-col product_id --rating-col rating --timestamp-col timestamp
  --test-size 0.2 --random-state 42
Labels should distinguish genuine/human reviews from deceptive/fake reviews.
"""
import argparse, json, os, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to dataset")
    p.add_argument("--text-col", default="review_text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--human-label", default="human", help="Exact dataset label meaning human/genuine")
    p.add_argument("--reviewer-col", default=None)
    p.add_argument("--product-col", default=None)
    p.add_argument("--rating-col", default=None)
    p.add_argument("--timestamp-col", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-features", type=int, default=50000)
    return p.parse_args()

def linguistic_features(text):
    text = "" if pd.isna(text) else str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    return {
        "char_count": len(text),
        "word_count": len(words),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "sentence_count": len(sentences),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "uppercase_ratio": sum(c.isupper() for c in text) / max(1, sum(c.isalpha() for c in text)),
        "punctuation_ratio": sum(not c.isalnum() and not c.isspace() for c in text) / max(1, len(text)),
        "unique_word_ratio": len(set(words)) / max(1, len(words)),
    }

def build_features(df, args):
    texts=df[args.text_col].fillna("").astype(str)
    ling=pd.DataFrame([linguistic_features(t) for t in texts], index=df.index)
    numeric=ling.copy()
    if args.rating_col and args.rating_col in df:
        numeric["rating"] = pd.to_numeric(df[args.rating_col], errors="coerce")
    # Behavioural features are calculated using available metadata; reviewer-level
    # statistics are computed from the whole unlabeled feature table, not target values.
    if args.reviewer_col and args.reviewer_col in df:
        ids=df[args.reviewer_col].fillna("__missing__").astype(str)
        counts=ids.value_counts()
        numeric["reviewer_review_count"] = ids.map(counts).astype(float)
    if args.product_col and args.product_col in df:
        ids=df[args.product_col].fillna("__missing__").astype(str)
        counts=ids.value_counts()
        numeric["product_review_count"] = ids.map(counts).astype(float)
    if args.timestamp_col and args.timestamp_col in df:
        ts=pd.to_datetime(df[args.timestamp_col], errors="coerce", utc=True)
        numeric["timestamp_hour"] = ts.dt.hour
        numeric["timestamp_dayofweek"] = ts.dt.dayofweek
        numeric["timestamp_epoch_days"] = ts.astype("int64", errors="ignore") / 86400000000000
        numeric.loc[ts.isna(), ["timestamp_hour","timestamp_dayofweek","timestamp_epoch_days"]] = np.nan
        if args.reviewer_col and args.reviewer_col in df:
            tmp=pd.DataFrame({"reviewer":df[args.reviewer_col].fillna("__missing__").astype(str),"ts":ts})
            tmp["prev_ts"]=tmp.groupby("reviewer")["ts"].shift(1)
            delta=(tmp["ts"]-tmp["prev_ts"]).dt.total_seconds()/3600
            numeric["hours_since_previous_reviewer_review"] = delta
    numeric=numeric.replace([np.inf,-np.inf],np.nan)
    return texts, numeric

def main():
    args = parse_args()
    path = Path(args.data)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    for col in (args.text_col,args.label_col):
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing. Available columns: {list(df.columns)}")
    df=df.dropna(subset=[args.label_col]).copy()
    df[args.label_col]=df[args.label_col].astype(str).str.strip()
    human_label=str(args.human_label).strip()
    # Binary mapping: human label -> 1; all other labels -> 0.
    y=(df[args.label_col].str.casefold()==human_label.casefold()).astype(int)
    if y.nunique()!=2:
        raise ValueError(
            f"Expected both human label '{human_label}' and non-human labels in '{args.label_col}'. "
            f"Observed labels: {sorted(df[args.label_col].unique().tolist())}"
        )
    texts, numeric=build_features(df,args)
    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        texts, numeric, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    vectorizer=TfidfVectorizer(ngram_range=(1,2), max_features=args.max_features,
                               min_df=2, strip_accents="unicode", sublinear_tf=True)
    Xtr_text=vectorizer.fit_transform(X_train_text)
    Xte_text=vectorizer.transform(X_test_text)
    imputer=SimpleImputer(strategy="median", add_indicator=True)
    scaler=StandardScaler()
    Xtr_num=imputer.fit_transform(X_train_num)
    Xte_num=imputer.transform(X_test_num)
    Xtr_num=scaler.fit_transform(Xtr_num)
    Xte_num=scaler.transform(Xte_num)
    Xtr=hstack([Xtr_text, csr_matrix(Xtr_num)]).tocsr()
    Xte=hstack([Xte_text, csr_matrix(Xte_num)]).tocsr()
    model=LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)
    model.fit(Xtr,y_train)
    pred=model.predict(Xte)
    prob=model.predict_proba(Xte)[:,1]
    metrics={
      "rows_total": int(len(df)), "train_rows": int(len(y_train)), "test_rows": int(len(y_test)),
      "human_label": human_label, "observed_labels": sorted(df[args.label_col].unique().tolist()),
      "accuracy": float(accuracy_score(y_test,pred)),
      "classification_report": classification_report(y_test,pred,target_names=["deceptive_or_nonhuman","human"],output_dict=True,zero_division=0),
      "confusion_matrix_labels_order": ["deceptive_or_nonhuman","human"],
      "confusion_matrix": confusion_matrix(y_test,pred,labels=[0,1]).tolist(),
      "roc_auc": float(roc_auc_score(y_test,prob)) if y_test.nunique()==2 else None,
      "feature_columns": list(numeric.columns),
      "text_column": args.text_col, "label_column": args.label_col
    }
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    joblib.dump({"vectorizer":vectorizer,"imputer":imputer,"scaler":scaler,"model":model,
                 "numeric_columns":list(numeric.columns),"config":{
                   "text_col":args.text_col,"human_label":human_label,
                   "reviewer_col":args.reviewer_col,"product_col":args.product_col,
                   "rating_col":args.rating_col,"timestamp_col":args.timestamp_col
                 }}, ARTIFACT_DIR/"review_model.joblib")
    (ARTIFACT_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    print(f"\nSaved model: {ARTIFACT_DIR/'review_model.joblib'}")

if __name__=="__main__": main()
