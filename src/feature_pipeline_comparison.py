import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.features import extract

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df.drop_duplicates(subset=["URL"]).copy()
df["ClassLabel"] = df["ClassLabel"].astype(int)

original_cols = [c for c in df.columns if c not in ["URL", "ClassLabel"]]
X_original = df[original_cols].apply(pd.to_numeric, errors="coerce")

extracted = pd.DataFrame(
    [extract(str(url)) for url in df["URL"]],
    index=df.index
)

valid = X_original.notna().all(axis=1) & extracted.notna().all(axis=1)
df = df.loc[valid]
X_original = X_original.loc[valid]
X_extracted = extracted.loc[valid]
y = df["ClassLabel"]

train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42, stratify=y
)

results = []

for name, X in [
    ("Original 16 precomputed features", X_original),
    ("Extracted 29 lexical features", X_extracted)
]:
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X.loc[train_idx], y.loc[train_idx])
    pred = model.predict(X.loc[test_idx])
    prob = model.predict_proba(X.loc[test_idx])[:, 1]

    results.append({
        "feature_set": name,
        "features": X.shape[1],
        "accuracy": accuracy_score(y.loc[test_idx], pred),
        "precision_class1": precision_score(y.loc[test_idx], pred, zero_division=0),
        "recall_class1": recall_score(y.loc[test_idx], pred, zero_division=0),
        "f1_class1": f1_score(y.loc[test_idx], pred, zero_division=0),
        "roc_auc": roc_auc_score(y.loc[test_idx], prob)
    })

result = pd.DataFrame(results)
result.to_csv("results/feature_pipeline_comparison.csv", index=False)

print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results/feature_pipeline_comparison.csv")
print("Class 1 is the positive class, using the dataset's original labels.")
