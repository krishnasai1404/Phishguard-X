import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, confusion_matrix
)
from src.features import extract

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"]).drop_duplicates("URL").copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url").copy()
external["label"] = external["label"].str.strip().str.lower()
external = external[external["label"].isin(["phishing", "legitimate"])].copy()
y_ext = external["label"].map({"phishing": 0, "legitimate": 1}).to_numpy()

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = original["ClassLabel"].reset_index(drop=True)
X_ext = pd.DataFrame([extract(u) for u in external["url"].astype(str)])

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

removed = ["query_len", "n_digits", "n_percent", "https"]
X_train = X_train.drop(columns=removed)
X_ext = X_ext.drop(columns=removed)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

idx = list(model.classes_).index(1)
legitimate_score = model.predict_proba(X_ext)[:, idx]
phishing_score = 1 - legitimate_score
pred_phishing = (phishing_score >= 0.5).astype(int)
true_phishing = (y_ext == 0).astype(int)

print("FEATURE-REMOVAL EXPERIMENT")
print("Removed:", ", ".join(removed))
print(f"Accuracy: {accuracy_score(true_phishing, pred_phishing):.4f}")
print(f"Phishing precision: {precision_score(true_phishing, pred_phishing, zero_division=0):.4f}")
print(f"Phishing recall: {recall_score(true_phishing, pred_phishing, zero_division=0):.4f}")
print(f"Phishing F1: {f1_score(true_phishing, pred_phishing, zero_division=0):.4f}")
print(f"Phishing average precision: {average_precision_score(true_phishing, phishing_score):.4f}")
print("Confusion matrix (rows=true [legitimate, phishing], columns=predicted [legitimate, phishing]):")
print(confusion_matrix(1 - true_phishing, 1 - pred_phishing))

pd.DataFrame([{
    "removed_features": ", ".join(removed),
    "accuracy": accuracy_score(true_phishing, pred_phishing),
    "phishing_precision": precision_score(true_phishing, pred_phishing, zero_division=0),
    "phishing_recall": recall_score(true_phishing, pred_phishing, zero_division=0),
    "phishing_f1": f1_score(true_phishing, pred_phishing, zero_division=0),
    "phishing_average_precision": average_precision_score(true_phishing, phishing_score)
}]).to_csv("results/feature_removal_external.csv", index=False)

print("Saved results/feature_removal_external.csv")
