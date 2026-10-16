import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("results/external_predictions.csv")
y = df["true_label"].astype(int)
scores = df["phishing_score"].astype(float)

print("Threshold | Phishing precision | Phishing recall | F1 phishing | Legitimate recall")
rows = []

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (scores >= threshold).astype(int)
    pp = precision_score(y, pred, zero_division=0)
    pr = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    lr = recall_score(1 - y, 1 - pred, zero_division=0)

    rows.append({
        "threshold": threshold,
        "phishing_precision": pp,
        "phishing_recall": pr,
        "f1_phishing": f1,
        "legitimate_recall": lr
    })

    print(f"{threshold:9.2f} | {pp:18.4f} | {pr:15.4f} | {f1:11.4f} | {lr:17.4f}")

pd.DataFrame(rows).to_csv("results/external_threshold_diagnostic_corrected.csv", index=False)
print("\nSaved results/external_threshold_diagnostic_corrected.csv")
