import pandas as pd
from src.features import extract

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"])
original = original.drop_duplicates(subset=["URL"]).copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)

original_features = pd.DataFrame(
    [extract(u) for u in original["URL"].astype(str)]
)
original_features["dataset"] = "Original"
original_features["label"] = original["ClassLabel"].map({
    0: "phishing",
    1: "legitimate"
}).to_numpy()

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"])
external = external.drop_duplicates(subset=["url"]).copy()
external["label"] = external["label"].astype(str).str.strip().str.lower()
external = external[
    external["label"].isin(["phishing", "legitimate"])
].copy()

external_features = pd.DataFrame(
    [extract(u) for u in external["url"].astype(str)]
)
external_features["dataset"] = "External"
external_features["label"] = external["label"].to_numpy()

combined = pd.concat(
    [original_features, external_features],
    ignore_index=True
)

feature_cols = [
    c for c in combined.select_dtypes(include="number").columns
]

summary = combined.groupby(["dataset", "label"])[feature_cols].mean().T
summary.to_csv("results/dataset_feature_distribution.csv")

print("Mean feature values by dataset and class:")
print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
print("\nSaved results/dataset_feature_distribution.csv")

print("\nURL counts by dataset and class:")
print(combined.groupby(["dataset", "label"]).size().to_string())
