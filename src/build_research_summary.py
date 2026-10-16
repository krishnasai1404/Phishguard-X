import pandas as pd
from pathlib import Path

def markdown_table(df, digits=4):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "")
    headers = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, divider] + rows)

sections = []

ablation = pd.read_csv("results/fair_feature_ablation.csv")
sections.append("## 1. Same-split feature ablation\n\n"
    + markdown_table(ablation))

domain = pd.read_csv("results/domain_generalization.csv")
domain_cols = [
    "train_urls", "test_urls", "train_hosts", "test_hosts",
    "host_overlap", "grouped_accuracy", "grouped_roc_auc"
]
sections.append("## 2. Host-grouped holdout\n\n"
    + markdown_table(domain[[c for c in domain_cols if c in domain.columns]]))

ci = pd.read_csv("results/external_metrics_with_ci.csv")
sections.append("## 3. External validation with bootstrap confidence intervals\n\n"
    + markdown_table(ci))

thresholds = pd.read_csv("results/external_threshold_summary.csv")
sections.append("## 4. External threshold analysis\n\n"
    + markdown_table(thresholds))

report = """# PhishGuard-X Research Results Summary

Generated from the project's saved experiment outputs.

## Interpretation notes

- The feature-ablation models use the same training split and external dataset.
- The host-grouped holdout is a separate experiment.
- External metrics from different model runs must not be treated as directly comparable.
- Bootstrap confidence intervals describe sampling uncertainty on this dataset, not guaranteed real-world performance.
- External threshold results are diagnostic; choose the final threshold using validation data.
- The external evaluation shows a substantial false-positive problem that needs further work.

""" + "\n\n".join(sections) + "\n"

path = Path("results/research_results_summary.md")
path.write_text(report, encoding="utf-8")
print(f"Created {path}")
print(f"Report size: {path.stat().st_size} bytes")
print("\nSections included:")
for section in sections:
    print(section.splitlines()[0])
