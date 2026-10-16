import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from src.features import extract

def get_host(url):
    try:
        value = url if "://" in str(url) else "http://" + str(url)
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except Exception:
        return ""

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"]).drop_duplicates("URL").copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)
original["host"] = original["URL"].astype(str).map(get_host)
original = original[original["host"] != ""].reset_index(drop=True)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url").copy()
external["label"] = external["label"].str.strip().str.lower()
external = external[external["label"].isin(["phishing", "legitimate"])].copy()
external["target"] = external["label"].map({"phishing": 0, "legitimate": 1})
external["host"] = external["url"].astype(str).map(get_host)
external = external[external["host"] != ""].reset_index(drop=True)

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = original["ClassLabel"].reset_index(drop=True)
groups = original["host"].reset_index(drop=True)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

classes = list(model.classes_)
positive_index = classes.index(1)
test_scores = model.predict_proba(X_test)[:, positive_index]
test_pred = (test_scores >= 0.5).astype(int)

print("ORIGINAL DATA: HOST-GROUPED HOLDOUT")
print(f"Train URLs: {len(train_idx)}")
print(f"Test URLs: {len(test_idx)}")
print(f"Unique train hosts: {groups.iloc[train_idx].nunique()}")
print(f"Unique test hosts: {groups.iloc[test_idx].nunique()}")
print(f"Host overlap: {len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))}")
print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, test_scores):.4f}")
print(f"Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")

print("\nEXTERNAL DATA: URL-LEVEL EVALUATION")
X_ext = pd.DataFrame([extract(u) for u in external["url"].astype(str)])
y_ext = external["target"].to_numpy()
ext_scores = 1 - model.predict_proba(X_ext)[:, positive_index]
ext_pred = (ext_scores >= 0.5).astype(int)

print(f"External URLs: {len(external)}")
print(f"External unique hosts: {external['host'].nunique()}")
print(f"Accuracy: {accuracy_score(y_ext, 1 - ext_pred):.4f}")
print(f"Phishing precision: {precision_score(1-y_ext, ext_pred, zero_division=0):.4f}")
print(f"Phishing recall: {recall_score(1-y_ext, ext_pred, zero_division=0):.4f}")
print(f"Phishing F1: {f1_score(1-y_ext, ext_pred, zero_division=0):.4f}")
print(f"Phishing average precision: {average_precision_score(1-y_ext, ext_scores):.4f}")
print(f"Confusion matrix (true legitimate/phishing mapped to 1/0, see note below):\n{confusion_matrix(y_ext, 1-ext_pred)}")

pd.DataFrame([{
    "train_urls": len(train_idx),
    "test_urls": len(test_idx),
    "train_hosts": groups.iloc[train_idx].nunique(),
    "test_hosts": groups.iloc[test_idx].nunique(),
    "host_overlap": len(set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])),
    "grouped_accuracy": accuracy_score(y_test, test_pred),
    "grouped_roc_auc": roc_auc_score(y_test, test_scores),
    "external_accuracy": accuracy_score(y_ext, 1-ext_pred),
    "external_phishing_precision": precision_score(1-y_ext, ext_pred, zero_division=0),
    "external_phishing_recall": recall_score(1-y_ext, ext_pred, zero_division=0),
    "external_phishing_f1": f1_score(1-y_ext, ext_pred, zero_division=0),
    "external_phishing_ap": average_precision_score(1-y_ext, ext_scores),
}]).to_csv("results/domain_generalization.csv", index=False)

print("\nSaved results/domain_generalization.csv")
