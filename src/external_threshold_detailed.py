import pandas as pd
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, balanced_accuracy_score, accuracy_score
)

df = pd.read_csv("results/external_predictions.csv")
df = df.dropna(subset=["true_label", "phishing_score"])
y = df["true_label"].astype(int)
scores = df["phishing_score"].astype(float)

rows = []

print("Threshold | Precision | Phishing Recall | F1 | Specificity | FPR | FNR | Balanced Acc.")

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    precision = precision_score(y, pred, zero_division=0)
    phishing_recall = recall_score(y, pred, zero_division=0)
    legitimate_recall = tn / (tn + fp) if (tn + fp) else 0
    f1 = f1_score(y, pred, zero_division=0)
    fpr = fp / (fp + tn) if (fp + tn) else 0
    fnr = fn / (fn + tp) if (fn + tp) else 0
    balanced_acc = balanced_accuracy_score(y, pred)
    accuracy = accuracy_score(y, pred)

    row = {
        "threshold": threshold,
        "accuracy": accuracy,
        "phishing_precision": precision,
        "phishing_recall": phishing_recall,
        "f1_phishing": f1,
        "legitimate_recall_specificity": legitimate_recall,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "balanced_accuracy": balanced_acc,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    }
    rows.append(row)

    print(
        f"{threshold:9.2f} | {precision:.4f} | {phishing_recall:.4f} | "
        f"{f1:.4f} | {legitimate_recall:.4f} | {fpr:.4f} | "
        f"{fnr:.4f} | {balanced_acc:.4f}"
    )

out = pd.DataFrame(rows)
out.to_csv("results/external_threshold_detailed.csv", index=False)

print("\nSaved results/external_threshold_detailed.csv")
print("\nConfusion-matrix counts and all metrics:")
print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
