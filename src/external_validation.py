import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

FEATURES = [
    "url_length", "has_ip_address", "dot_count", "https_flag",
    "url_entropy", "token_count", "subdomain_count",
    "query_param_count", "tld_length", "path_length",
    "has_hyphen_in_domain", "number_of_digits", "tld_popularity",
    "suspicious_file_extension", "domain_name_length",
    "percentage_numeric_chars"
]

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"])
original = original.drop_duplicates(subset=["URL"])
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"])
external = external.drop_duplicates(subset=["url"])

print("Original label counts:")
print(original["ClassLabel"].value_counts().sort_index())
print("\nExternal label counts:")
print(external["label"].value_counts())

# Assumed mapping: original 0 = phishing, 1 = legitimate.
# Verify this mapping before interpreting the results.
label_map = {"phishing": 0, "legitimate": 1}
external["target"] = external["label"].str.strip().str.lower().map(label_map)

if external["target"].isna().any():
    raise ValueError("Unexpected external labels found.")

X = original[FEATURES].apply(pd.to_numeric, errors="coerce")
y = original["ClassLabel"]

if set(y.unique()) != {0, 1}:
    raise ValueError(f"Unexpected original labels: {sorted(y.unique())}")

X = X.replace([np.inf, -np.inf], np.nan)
valid = X.notna().all(axis=1)
X, y = X.loc[valid], y.loc[valid]

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# Import the project's original feature extractor.
from src.features import extract

X_ext = pd.DataFrame(
    [extract(url) for url in external["url"]],
    columns=FEATURES
).apply(pd.to_numeric, errors="coerce")

if list(X_ext.columns) != FEATURES:
    raise ValueError("External feature columns do not match original features.")

X_ext = X_ext.replace([np.inf, -np.inf], np.nan)
valid_ext = X_ext.notna().all(axis=1)
X_ext = X_ext.loc[valid_ext]
y_ext = external.loc[valid_ext, "target"].astype(int)

print("\nExternal rows evaluated:", len(y_ext))
print("External rows skipped:", len(external) - len(y_ext))

prob = model.predict_proba(X_ext)[:, list(model.classes_).index(1)]
pred = (prob >= 0.5).astype(int)

results = {
    "accuracy": accuracy_score(y_ext, pred),
    "precision": precision_score(y_ext, pred, zero_division=0),
    "recall": recall_score(y_ext, pred, zero_division=0),
    "f1": f1_score(y_ext, pred, zero_division=0),
    "roc_auc": roc_auc_score(y_ext, prob),
}

print("\nMetrics:")
for name, value in results.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion matrix (rows=true, columns=predicted; 0=phishing, 1=legitimate):")
print(confusion_matrix(y_ext, pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_ext, pred, labels=[0, 1],
    target_names=["phishing", "legitimate"], zero_division=0
))

pd.DataFrame([results]).to_csv(
    "results/external_validation.csv", index=False
)
print("\nSaved: results/external_validation.csv")
