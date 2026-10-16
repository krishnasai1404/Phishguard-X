import pandas as pd

ci = pd.read_csv("results/external_confidence_intervals.csv")
thresholds = pd.read_csv("results/external_threshold_diagnostic_corrected.csv")

ci.to_csv("results/external_metrics_with_ci.csv", index=False)
thresholds.to_csv("results/external_threshold_summary.csv", index=False)

print("External metrics with confidence intervals:")
print(ci.to_string(index=False))
print("\nCorrected threshold summary:")
print(thresholds.to_string(index=False))
print("\nSaved:")
print("results/external_metrics_with_ci.csv")
print("results/external_threshold_summary.csv")
