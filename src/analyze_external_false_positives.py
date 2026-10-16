import pandas as pd
from urllib.parse import urlparse

pred = pd.read_csv("results/external_predictions.csv")
external = pd.read_csv("data/external_balanced_urls.csv")

df = pred.merge(external[["url", "label"]].drop_duplicates("url"), on="url", how="left")
df["label"] = df["label"].str.strip().str.lower()
df["host"] = df["url"].apply(
    lambda u: (urlparse(u if "://" in str(u) else "http://" + str(u)).hostname or "").lower()
)

df["predicted_phishing"] = df["phishing_score"] >= 0.5
false_positives = df[
    (df["label"] == "legitimate") & df["predicted_phishing"]
].copy()

print("Legitimate URLs incorrectly flagged:", len(false_positives))
print("\nMost frequently flagged hosts:")
print(false_positives["host"].value_counts().head(20).to_string())

print("\nExamples of false positives:")
print(false_positives[["url", "phishing_score"]].head(15).to_string(index=False))

false_positives.to_csv("results/external_false_positives.csv", index=False)
print("\nSaved results/external_false_positives.csv")
