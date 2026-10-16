import pandas as pd
from pathlib import Path

files = {
    "fair_feature_ablation": "results/fair_feature_ablation.csv",
    "domain_generalization": "results/domain_generalization.csv",
    "external_metrics_with_ci": "results/external_metrics_with_ci.csv",
    "feature_removal_external": "results/feature_removal_external.csv",
}

frames = []

for experiment, filename in files.items():
    path = Path(filename)
    if path.exists():
        df = pd.read_csv(path)
        df.insert(0, "experiment", experiment)
        frames.append(df)
        print(f"Loaded {filename}: {len(df)} row(s)")
    else:
        print(f"Skipped missing file: {filename}")

if not frames:
    raise SystemExit("No experiment result files were found.")

combined = pd.concat(frames, ignore_index=True, sort=False)
output = Path("results/combined_experiment_results.csv")
combined.to_csv(output, index=False)

print(f"\nSaved {output}")
print(f"Total rows: {len(combined)}")
print("\nColumns:")
print(", ".join(combined.columns))
