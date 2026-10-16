import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.features import extract

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"]).drop_duplicates("URL").copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url").copy()
external["label"] = external["label"].str.strip().str.lower()
external = external[external["label"].isin(["phishing", "legitimate"])].copy()
external = external.reset_index(drop=True)

y_ext = external["label"].map({"phishing": 0, "legitimate": 1})

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = original["ClassLabel"].reset_index(drop=True)

X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_ext = pd.DataFrame(
    [extract(u) for u in external["url"].astype(str)]
).reset_index(drop=True)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

prob = model.predict_proba(X_ext)[:, list(model.classes_).index(1)]
y_phishing = (y_ext == 0).astype(int)
phishing_score = 1 - prob

precision, recall, _ = precision_recall_curve(y_phishing, phishing_score)
ap = average_precision_score(y_phishing, phishing_score)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision)
plt.xlabel("Phishing recall")
plt.ylabel("Phishing precision")
plt.title("External Precision-Recall Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/external_precision_recall.png", dpi=200)
plt.close()

pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
    "results/external_precision_recall.csv", index=False
)

pd.DataFrame({
    "url": external["url"],
    "true_label": y_phishing,
    "phishing_score": phishing_score
}).to_csv("results/external_predictions.csv", index=False)

print(f"External average precision: {ap:.4f}")
print("Saved results/external_precision_recall.png")
print("Saved results/external_precision_recall.csv")
print("Saved results/external_predictions.csv")
