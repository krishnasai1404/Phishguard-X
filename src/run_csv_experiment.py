
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

DATA_PATH = Path("data/url_features_extracted1.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["ClassLabel"])
df = df.drop_duplicates(subset=["URL"]).copy()

feature_cols = [c for c in df.columns if c not in ["URL", "ClassLabel"]]
X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
y = df["ClassLabel"].astype(int)

valid = X.notna().all(axis=1)
X = X.loc[valid]
y = y.loc[valid]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42
    ),
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 4))

    results.append({
        "model": name,
        "accuracy": model.score(X_test, y_test),
        "roc_auc": roc_auc_score(y_test, prob),
    })

pd.DataFrame(results).to_csv(
    RESULTS_DIR / "csv_baseline_results.csv", index=False
)

print("\nSaved results to results/csv_baseline_results.csv")
print("Feature columns:", feature_cols)
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
