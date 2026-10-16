import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

df = pd.read_csv("results/external_predictions.csv")
y = df["true_label"].to_numpy(dtype=int)
scores = df["phishing_score"].to_numpy(dtype=float)

threshold = 0.10
pred = (scores >= threshold).astype(int)

observed = {
    "Average precision": average_precision_score(y, scores),
    "Phishing precision at 0.10": precision_score(y, pred, zero_division=0),
    "Phishing recall at 0.10": recall_score(y, pred, zero_division=0),
}

rng = np.random.default_rng(42)
n = len(y)
iterations = 1000
samples = {name: [] for name in observed}

for i in range(iterations):
    idx = rng.integers(0, n, size=n)
    y_b = y[idx]
    s_b = scores[idx]
    p_b = (s_b >= threshold).astype(int)

    if np.unique(y_b).size < 2:
        continue

    samples["Average precision"].append(
        average_precision_score(y_b, s_b)
    )
    samples["Phishing precision at 0.10"].append(
        precision_score(y_b, p_b, zero_division=0)
    )
    samples["Phishing recall at 0.10"].append(
        recall_score(y_b, p_b, zero_division=0)
    )

rows = []
for name, value in observed.items():
    values = np.asarray(samples[name])
    low, high = np.percentile(values, [2.5, 97.5])
    rows.append({
        "metric": name,
        "estimate": value,
        "ci_95_lower": low,
        "ci_95_upper": high,
        "bootstrap_iterations": len(values),
    })

result = pd.DataFrame(rows)
result.to_csv("results/external_confidence_intervals.csv", index=False)
print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results/external_confidence_intervals.csv")
