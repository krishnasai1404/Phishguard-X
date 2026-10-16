import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from src.features import extract

SEED = 42

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"])
original = original.drop_duplicates(subset=["URL"]).copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"])
external = external.drop_duplicates(subset=["url"]).copy()

label_map = {"phishing": 0, "legitimate": 1}
external["target"] = external["label"].str.strip().str.lower().map(label_map)

if external["target"].isna().any():
    raise ValueError("Unexpected external label values.")

if set(original["ClassLabel"].unique()) != {0, 1}:
    raise ValueError("Original dataset must have labels 0 and 1.")

print("Assumed mapping: 0 = phishing, 1 = legitimate")
print("Original rows:", len(original))
print("External rows:", len(external))

# Extract identical 29 features from both datasets.
X_original = pd.DataFrame(
    [extract(u) for u in original["URL"].astype(str)]
)
X_external = pd.DataFrame(
    [extract(u) for u in external["url"].astype(str)]
)

# Align feature order explicitly.
X_external = X_external.reindex(columns=X_original.columns)

X_original = X_original.replace([np.inf, -np.inf], np.nan)
X_external = X_external.replace([np.inf, -np.inf], np.nan)

y_original = original["ClassLabel"].reset_index(drop=True)
X_original = X_original.reset_index(drop=True)
valid_train = X_original.notna().all(axis=1)
X_original = X_original.loc[valid_train]
y_original = y_original.loc[valid_train]

X_train, _, y_train, _ = train_test_split(
    X_original, y_original,
    test_size=0.2, random_state=SEED, stratify=y_original
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=SEED,
    n_jobs=-1
)
model.fit(X_train, y_train)

X_external = X_external.reset_index(drop=True)
y_external = external["target"].astype(int).reset_index(drop=True)
valid_test = X_external.notna().all(axis=1)
X_external = X_external.loc[valid_test]
y_external = y_external.loc[valid_test]

prob = model.predict_proba(X_external)[:, list(model.classes_).index(1)]
pred = (prob >= 0.5).astype(int)

results = {
    "model": "RandomForest_29_consistent_features",
    "train_rows": len(X_train),
    "external_test_rows": len(y_external),
    "accuracy": accuracy_score(y_external, pred),
    "precision_legitimate": precision_score(
        y_external, pred, pos_label=1, zero_division=0
    ),
    "recall_legitimate": recall_score(
        y_external, pred, pos_label=1, zero_division=0
    ),
    "f1_legitimate": f1_score(
        y_external, pred, pos_label=1, zero_division=0
    ),
    "roc_auc_legitimate": roc_auc_score(y_external, prob),
}

print("\nExternal validation metrics:")
for key, value in results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, (float, np.floating))
          else f"{key}: {value}")

print("\nConfusion matrix (rows=true, columns=predicted; 0=phishing, 1=legitimate):")
print(confusion_matrix(y_external, pred, labels=[0, 1]))

print("\nClassification report:")
print(classification_report(
    y_external, pred,
    labels=[0, 1],
    target_names=["phishing", "legitimate"],
    zero_division=0
))

pd.DataFrame([results]).to_csv(
    "results/external_validation_consistent.csv", index=False
)
print("\nSaved results/external_validation_consistent.csv")
