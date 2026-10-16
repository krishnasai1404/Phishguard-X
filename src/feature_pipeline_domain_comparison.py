import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from src.features import extract

def get_host(url):
    try:
        value = str(url)
        if "://" not in value:
            value = "http://" + value
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except Exception:
        return ""

def evaluate(y_true, scores):
    pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": accuracy_score(y_true, pred),
        "phishing_precision": precision_score(y_true, pred, zero_division=0),
        "phishing_recall": recall_score(y_true, pred, zero_division=0),
        "phishing_f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "phishing_average_precision": average_precision_score(y_true, scores),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df.drop_duplicates(subset=["URL"]).copy()
df["ClassLabel"] = df["ClassLabel"].astype(int)
df["host"] = df["URL"].astype(str).map(get_host)
df = df[df["host"] != ""].reset_index(drop=True)

precomputed_cols = [
    c for c in df.columns if c not in ["URL", "ClassLabel", "host"]
]
X_precomputed = df[precomputed_cols].apply(pd.to_numeric, errors="coerce")
X_extracted = pd.DataFrame(
    [extract(url) for url in df["URL"].astype(str)],
    index=df.index
)

valid = X_precomputed.notna().all(axis=1) & X_extracted.notna().all(axis=1)
df = df.loc[valid].reset_index(drop=True)
X_precomputed = X_precomputed.loc[valid].reset_index(drop=True)
X_extracted = X_extracted.loc[valid].reset_index(drop=True)

y = (df["ClassLabel"] == 0).astype(int)
groups = df["host"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(
    splitter.split(X_precomputed, y, groups=groups)
)

print("HOST-GROUPED FEATURE PIPELINE COMPARISON")
print(f"Train URLs: {len(train_idx)}")
print(f"Test URLs: {len(test_idx)}")
print(f"Train hosts: {groups.iloc[train_idx].nunique()}")
print(f"Test hosts: {groups.iloc[test_idx].nunique()}")
print(f"Host overlap: {len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))}")
print("Positive class: phishing (ClassLabel = 0)")
print("Threshold: phishing score >= 0.5")

rows = []

for name, X in [
    ("Original 16 precomputed features", X_precomputed),
    ("Extracted 29 lexical features", X_extracted)
]:
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    phishing_index = list(model.classes_).index(1)
    scores = model.predict_proba(X.iloc[test_idx])[:, phishing_index]
    result = evaluate(y.iloc[test_idx].to_numpy(), scores)

    print(f"\n{name}")
    print(result)

    rows.append({
        "feature_set": name,
        "feature_count": X.shape[1],
        "train_urls": len(train_idx),
        "test_urls": len(test_idx),
        "host_overlap": len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])),
        **result
    })

pd.DataFrame(rows).to_csv(
    "results/feature_pipeline_domain_comparison.csv", index=False
)
print("\nSaved results/feature_pipeline_domain_comparison.csv")
print("External evaluation omitted: the original 16 feature definitions")
print("cannot be reproduced for external URLs from the available files.")
