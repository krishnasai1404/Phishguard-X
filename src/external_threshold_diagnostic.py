import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src.features import extract
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"]).drop_duplicates("URL")
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url")
y_ext = external["label"].str.lower().map({"phishing": 0, "legitimate": 1}).reset_index(drop=True)

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = original["ClassLabel"].reset_index(drop=True)
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_ext = pd.DataFrame([extract(u) for u in external["url"].astype(str)]).reset_index(drop=True)
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
prob = model.predict_proba(X_ext)[:, list(model.classes_).index(1)]

print("Threshold | Phishing recall | Legitimate recall | Legitimate precision | F1 phishing")
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (prob >= threshold).astype(int)
    phishing_recall = recall_score(y_ext, pred, pos_label=0, zero_division=0)
    legitimate_recall = recall_score(y_ext, pred, pos_label=1, zero_division=0)
    legitimate_precision = precision_score(y_ext, pred, pos_label=1, zero_division=0)
    f1_phishing = f1_score(y_ext, pred, pos_label=0, zero_division=0)
    print(f"{threshold:>9.2f} | {phishing_recall:>15.4f} | {legitimate_recall:>17.4f} | {legitimate_precision:>20.4f} | {f1_phishing:>11.4f}")
