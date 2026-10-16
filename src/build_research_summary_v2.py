from pathlib import Path
import pandas as pd

OUT = Path("results/research_results_summary_v2.md")

def load(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else None

def table(df, columns=None, headers=None, limit=None):
    if df is None or df.empty:
        return "_Results file not available._\n"

    d = df.copy()
    if columns:
        d = d[[c for c in columns if c in d.columns]]
    if limit:
        d = d.head(limit)

    if headers:
        d = d.rename(columns=headers)

    cols = list(d.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |"
    ]

    for _, row in d.iterrows():
        vals = []
        for value in row:
            if pd.isna(value):
                vals.append("")
            elif isinstance(value, float):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"

parts = [
    "# PhishGuard-X Research Results Summary",
    "",
    "## Executive summary",
    "",
    "The experiments show excellent performance on the original dataset's "
    "host-grouped holdout, but substantially weaker performance on an external "
    "dataset. This gap indicates that strong in-dataset results do not establish "
    "reliable real-world generalization.",
    "",
    "The external evaluation has a substantial false-positive problem. Raising "
    "the decision threshold improves precision and legitimate-URL specificity "
    "but reduces phishing recall.",
    "",
    "## Evaluation and interpretation notes",
    "",
    "- Phishing is the positive class in the external evaluation.",
    "- The host-grouped experiment and external evaluation are separate settings.",
    "- Results from different model runs must not be compared as if they came "
    "from one identical experiment.",
    "- The original dataset's precomputed feature definitions are not fully "
    "documented, so they cannot be reproduced consistently for external URLs.",
    "- Bootstrap confidence intervals are URL-level estimates and may understate "
    "uncertainty when URLs share hosts or data sources.",
    "- Thresholds tested on the external dataset are diagnostic; final threshold "
    "selection should use validation data, followed by an untouched test set.",
    "- The distribution-shift analysis identifies differences between datasets "
    "but does not prove that individual features cause the performance gap.",
    "",
    "## 1. Same-split feature ablation",
    ""
]

d = load("results/fair_feature_ablation.csv")
parts.append(table(
    d,
    ["model", "accuracy", "phishing_precision",
     "phishing_recall", "phishing_f1",
     "phishing_average_precision"],
    {
        "model": "Model",
        "accuracy": "Accuracy",
        "phishing_precision": "Phishing precision",
        "phishing_recall": "Phishing recall",
        "phishing_f1": "Phishing F1",
        "phishing_average_precision": "Average precision"
    }
))
parts.append("")

parts += [
    "",
    "## 2. Host-grouped holdout: feature-pipeline comparison",
    "",
    "The split keeps hosts disjoint between training and testing. Both feature "
    "pipelines use the same split and Random Forest settings.",
    ""
]

d = load("results/feature_pipeline_domain_comparison.csv")
parts.append(table(
    d,
    ["feature_set", "feature_count", "accuracy", "phishing_precision",
     "phishing_recall", "phishing_f1", "roc_auc",
     "phishing_average_precision", "host_overlap"],
    {"feature_set": "Feature pipeline", "feature_count": "Features",
     "accuracy": "Accuracy", "phishing_precision": "Phishing precision",
     "phishing_recall": "Phishing recall", "phishing_f1": "Phishing F1",
     "roc_auc": "ROC-AUC",
     "phishing_average_precision": "Average precision",
     "host_overlap": "Host overlap"}
))

parts += [
    "",
    "The near-perfect grouped scores apply to this dataset and split only. They "
    "do not override the weaker external results.",
    "",
    "## 3. External validation with bootstrap confidence intervals",
    ""
]

d = load("results/external_metrics_with_ci.csv")
parts.append(table(d))

parts += [
    "",
    "## 4. External feature-group ablation",
    "",
    "These are diagnostic experiments on the external dataset. Removing a feature "
    "group changes multiple metrics, and the best result depends on the chosen "
    "objective. The full feature set has the highest average precision among "
    "these tested ablations.",
    ""
]

d = load("results/feature_group_ablation_external.csv")
parts.append(table(
    d,
    ["experiment", "feature_count", "accuracy", "phishing_precision",
     "phishing_recall", "phishing_f1", "roc_auc", "average_precision"],
    {"experiment": "Experiment", "feature_count": "Features",
     "accuracy": "Accuracy", "phishing_precision": "Phishing precision",
     "phishing_recall": "Phishing recall", "phishing_f1": "Phishing F1",
     "roc_auc": "ROC-AUC", "average_precision": "Average precision"}
))

parts += [
    "",
    "## 5. External threshold analysis",
    "",
    "Higher thresholds reduce false positives but increase false negatives. "
    "No threshold should be declared final using this evaluation set alone.",
    ""
]

d = load("results/external_threshold_detailed.csv")
parts.append(table(
    d,
    ["threshold", "accuracy", "phishing_precision", "phishing_recall",
     "f1_phishing", "legitimate_recall_specificity", "false_positive_rate",
     "false_negative_rate", "balanced_accuracy", "TN", "FP", "FN", "TP"],
    {"threshold": "Threshold", "accuracy": "Accuracy",
     "phishing_precision": "Phishing precision",
     "phishing_recall": "Phishing recall", "f1_phishing": "Phishing F1",
     "legitimate_recall_specificity": "Specificity",
     "false_positive_rate": "FPR", "false_negative_rate": "FNR",
     "balanced_accuracy": "Balanced accuracy",
     "TN": "TN", "FP": "FP", "FN": "FN", "TP": "TP"}
))

parts += [
    "",
    "## 6. Dataset distribution shift",
    "",
    "The following table lists the largest standardized mean differences for "
    "each class. These are descriptive comparisons, not causal findings.",
    ""
]

d = load("results/distribution_shift_ranking.csv")
if d is not None:
    for label in ["phishing", "legitimate"]:
        subset = d[d["label"] == label].sort_values(
            "absolute_difference", ascending=False
        ).head(10)
        parts += [
            f"### {label.title()} URLs",
            "",
            table(
                subset,
                ["feature", "original_mean", "external_mean",
                 "standardized_difference", "absolute_difference"],
                {"feature": "Feature", "original_mean": "Original mean",
                 "external_mean": "External mean",
                 "standardized_difference": "Standardized difference",
                 "absolute_difference": "Absolute difference"}
            ),
            ""
        ]
else:
    parts.append("_Distribution-shift file not available._\n")

parts += [
    "## 7. Conclusions and limitations",
    "",
    "1. The host-grouped holdout results are exceptionally high on the original "
    "dataset, but external performance is much lower.",
    "2. The external model has high phishing recall at lower thresholds, at the "
    "cost of many legitimate URLs being flagged.",
    "3. Feature-group ablation offers diagnostic evidence, but no result here "
    "establishes a universally best feature set.",
    "4. Distribution differences suggest substantial dataset shift. Further "
    "work should investigate collection sources, label quality, temporal "
    "generalization, calibration, and threshold selection.",
    "5. Future claims should be based on a validation procedure that separates "
    "model selection from final testing, ideally with a separate time-based or "
    "source-diverse dataset.",
    "",
    "## Reproducibility",
    "",
    "This report is assembled from the project's saved CSV experiment outputs. "
    "See the corresponding files in `results/` for the underlying measurements.",
    ""
]

OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"Created {OUT}")
print(f"Report size: {OUT.stat().st_size} bytes")

