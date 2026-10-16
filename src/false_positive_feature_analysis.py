import pandas as pd
from src.features import extract

pred = pd.read_csv("results/external_predictions.csv")
external = pd.read_csv("data/external_balanced_urls.csv")
external = external.drop_duplicates("url").copy()
external["label"] = external["label"].str.strip().str.lower()

df = pred.merge(external[["url", "label"]], on="url", how="left")
df = df[df["label"] == "legitimate"].copy()

df["predicted_phishing"] = df["phishing_score"] >= 0.5

features = pd.DataFrame([extract(u) for u in df["url"].astype(str)])
features["prediction_group"] = df["predicted_phishing"].map({
    True: "False positive",
    False: "Correctly classified legitimate"
}).to_numpy()

summary = features.groupby("prediction_group").mean(numeric_only=True).T
summary["difference"] = (
    summary["False positive"] -
    summary["Correctly classified legitimate"]
)
summary = summary.sort_values("difference", key=abs, ascending=False)

print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
summary.to_csv("results/false_positive_feature_comparison.csv")

print("\nSaved results/false_positive_feature_comparison.csv")
