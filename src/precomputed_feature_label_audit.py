import pandas as pd
from sklearn.metrics import roc_auc_score

path = "data/url_features_extracted1.csv"
df = pd.read_csv(path)
df = df.dropna(subset=["URL", "ClassLabel"]).drop_duplicates(subset=["URL"]).copy()
df["ClassLabel"] = df["ClassLabel"].astype(int)

feature_cols = [c for c in df.columns if c not in ["URL", "ClassLabel"]]
rows = []

for col in feature_cols:
    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.notna()
    if valid.sum() == 0 or df.loc[valid, "ClassLabel"].nunique() < 2:
        continue

    auc = roc_auc_score(df.loc[valid, "ClassLabel"], values[valid])
    rows.append({
        "feature": col,
        "valid_rows": int(valid.sum()),
        "class0_mean": values[valid & (df["ClassLabel"] == 0)].mean(),
        "class1_mean": values[valid & (df["ClassLabel"] == 1)].mean(),
        "auc": auc,
        "direction_independent_auc": max(auc, 1 - auc)
    })

result = pd.DataFrame(rows).sort_values(
    "direction_independent_auc", ascending=False
)
result.to_csv("results/precomputed_feature_label_audit.csv", index=False)

print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results/precomputed_feature_label_audit.csv")
