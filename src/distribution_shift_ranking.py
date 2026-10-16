import pandas as pd
import numpy as np
from src.features import extract

def load_data():
    original = pd.read_csv("data/url_features_extracted1.csv")
    original = original.dropna(subset=["URL", "ClassLabel"])
    original = original.drop_duplicates(subset=["URL"]).copy()
    original["label"] = original["ClassLabel"].astype(int).map({
        0: "phishing", 1: "legitimate"
    })
    original["dataset"] = "Original"
    original["url"] = original["URL"].astype(str)

    external = pd.read_csv("data/external_balanced_urls.csv")
    external = external.dropna(subset=["url", "label"])
    external = external.drop_duplicates(subset=["url"]).copy()
    external["label"] = external["label"].astype(str).str.strip().str.lower()
    external = external[external["label"].isin(["phishing", "legitimate"])].copy()
    external["dataset"] = "External"
    external["url"] = external["url"].astype(str)

    return pd.concat([
        original[["url", "label", "dataset"]],
        external[["url", "label", "dataset"]]
    ], ignore_index=True)

df = load_data()
features = pd.DataFrame([extract(u) for u in df["url"]])
features["label"] = df["label"].to_numpy()
features["dataset"] = df["dataset"].to_numpy()

rows = []
for label in ["phishing", "legitimate"]:
    a = features[(features["dataset"] == "Original") & (features["label"] == label)]
    b = features[(features["dataset"] == "External") & (features["label"] == label)]

    for col in features.select_dtypes(include="number").columns:
        mean_a, mean_b = a[col].mean(), b[col].mean()
        var_a, var_b = a[col].var(), b[col].var()
        pooled = np.sqrt((var_a + var_b) / 2)
        d = (mean_b - mean_a) / pooled if pooled > 0 else 0

        rows.append({
            "label": label,
            "feature": col,
            "original_mean": mean_a,
            "external_mean": mean_b,
            "standardized_difference": d,
            "absolute_difference": abs(d)
        })

result = pd.DataFrame(rows).sort_values(
    ["label", "absolute_difference"], ascending=[True, False]
)
result.to_csv("results/distribution_shift_ranking.csv", index=False)

for label in ["phishing", "legitimate"]:
    print(f"\n{label.upper()} — LARGEST DISTRIBUTION SHIFTS")
    print(result[result["label"] == label].head(10).to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

print("\nSaved results/distribution_shift_ranking.csv")
