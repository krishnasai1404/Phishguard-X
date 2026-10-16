
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"]).drop_duplicates(subset=["URL"]).copy()

df["host"] = df["URL"].astype(str).map(
    lambda u: (urlparse(u if "://" in u else "http://" + u).hostname or "").lower()
)
df = df[df["host"] != ""].copy()

features = [
    c for c in df.columns
    if c not in ["URL", "ClassLabel", "host"]
]
X = df[features].apply(pd.to_numeric, errors="coerce")
valid = X.notna().all(axis=1)
df = df.loc[valid].copy()
X = X.loc[valid]
y = df["ClassLabel"].astype(int)
groups = df["host"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

print("Train rows:", len(train_idx))
print("Test rows:", len(test_idx))
print("Overlapping hostnames:", len(
    set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
))
print("Training label counts:", y_train.value_counts().to_dict())
print("Test label counts:", y_test.value_counts().to_dict())

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name}: unseen-host test ===")
    print(classification_report(y_test, pred, zero_division=0))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

    results.append({
        "model": name,
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, prob),
    })

Path("results").mkdir(exist_ok=True)
pd.DataFrame(results).to_csv(
    "results/grouped_baseline_results.csv", index=False
)
print("\nSaved results/grouped_baseline_results.csv")
