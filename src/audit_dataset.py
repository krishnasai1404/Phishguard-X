
import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df.drop_duplicates(subset=["URL"])

y = df["ClassLabel"].astype(int)
features = [c for c in df.columns if c not in ["URL", "ClassLabel"]]

print("Label counts:")
print(y.value_counts().sort_index())
print("\nSingle-feature audit:")

rows = []

for col in features:
    x = pd.to_numeric(df[col], errors="coerce")
    valid = x.notna()
    x, target = x[valid], y[valid]

    if x.nunique() < 2 or target.nunique() < 2:
        continue

    auc = roc_auc_score(target, x)
    rows.append({
        "feature": col,
        "unique_values": x.nunique(),
        "auc": auc,
        "direction_independent_auc": max(auc, 1 - auc),
        "correlation_with_label": x.corr(target),
    })

result = pd.DataFrame(rows).sort_values(
    "direction_independent_auc", ascending=False
)

print(result.round(4).to_string(index=False))
result.to_csv("results/feature_audit.csv", index=False)
print("\nSaved results/feature_audit.csv")
