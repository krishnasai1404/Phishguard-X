import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score
)
from src.features import extract

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"]).drop_duplicates("URL").copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url").copy()
external["label"] = external["label"].str.strip().str.lower()
external = external[external["label"].isin(["phishing", "legitimate"])].copy()

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = original["ClassLabel"].reset_index(drop=True)
X_ext = pd.DataFrame([extract(u) for u in external["url"].astype(str)])
y_ext = (external["label"] == "phishing").astype(int).to_numpy()

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

removed = ["query_len", "n_digits", "n_percent", "https"]
rows = []

for name, columns in [
    ("All 29 features", list(X.columns)),
    ("Reduced features", [c for c in X.columns if c not in removed])
]:
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train[columns], y_train)

    legitimate_index = list(model.classes_).index(1)
    phishing_score = 1 - model.predict_proba(X_ext[columns])[:, legitimate_index]
    pred = (phishing_score >= 0.5).astype(int)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_ext, pred),
        "phishing_precision": precision_score(y_ext, pred, zero_division=0),
        "phishing_recall": recall_score(y_ext, pred, zero_division=0),
        "phishing_f1": f1_score(y_ext, pred, zero_division=0),
        "phishing_average_precision": average_precision_score(y_ext, phishing_score)
    }
    rows.append(row)
    print("\n", name)
    for key, value in row.items():
        if key != "model":
            print(f"{key}: {value:.4f}")

pd.DataFrame(rows).to_csv("results/fair_feature_ablation.csv", index=False)
print("\nSaved results/fair_feature_ablation.csv")
