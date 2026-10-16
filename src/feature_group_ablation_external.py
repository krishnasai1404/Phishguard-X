import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score
)
from urllib.parse import urlparse
from src.features import extract, FEATURE_GROUPS

def get_host(url):
    try:
        value = str(url)
        if "://" not in value:
            value = "http://" + value
        return (urlparse(value).hostname or "").lower().rstrip(".")
    except Exception:
        return ""

def evaluate(y, scores):
    pred = (scores >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "phishing_precision": precision_score(y, pred, zero_division=0),
        "phishing_recall": recall_score(y, pred, zero_division=0),
        "phishing_f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores),
        "average_precision": average_precision_score(y, scores)
    }

original = pd.read_csv("data/url_features_extracted1.csv")
original = original.dropna(subset=["URL", "ClassLabel"])
original = original.drop_duplicates(subset=["URL"]).copy()
original["ClassLabel"] = original["ClassLabel"].astype(int)
original["host"] = original["URL"].astype(str).map(get_host)
original = original[original["host"] != ""].reset_index(drop=True)

X = pd.DataFrame([extract(u) for u in original["URL"].astype(str)])
y = (original["ClassLabel"] == 0).astype(int)
groups = original["host"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, _ = next(splitter.split(X, y, groups=groups))

external = pd.read_csv("data/external_balanced_urls.csv")
external = external.dropna(subset=["url", "label"]).drop_duplicates("url").copy()
external["label"] = external["label"].astype(str).str.strip().str.lower()
external = external[external["label"].isin(["phishing", "legitimate"])].copy()
external["target"] = (external["label"] == "phishing").astype(int)
external["host"] = external["url"].astype(str).map(get_host)
external = external[external["host"] != ""].reset_index(drop=True)

X_ext = pd.DataFrame([extract(u) for u in external["url"].astype(str)])
y_ext = external["target"].to_numpy()

all_features = list(X.columns)
experiments = [("All 29 features", all_features)]

for group_name, group_features in FEATURE_GROUPS.items():
    experiments.append((f"Only {group_name}", group_features))
    remaining = [f for f in all_features if f not in group_features]
    experiments.append((f"Without {group_name}", remaining))

rows = []

print("FEATURE-GROUP ABLATION")
print(f"Training URLs: {len(train_idx)}")
print(f"External URLs: {len(external)}")
print(f"External unique hosts: {external['host'].nunique()}")
print("Positive class: phishing (1), threshold: 0.5")

for name, cols in experiments:
    model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    model.fit(X.iloc[train_idx][cols], y.iloc[train_idx])
    phishing_index = list(model.classes_).index(1)
    scores = model.predict_proba(X_ext[cols])[:, phishing_index]
    result = evaluate(y_ext, scores)

    print(f"\n{name} ({len(cols)} features)")
    print(result)

    rows.append({
        "experiment": name,
        "feature_count": len(cols),
        **result
    })

result_df = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
result_df.to_csv("results/feature_group_ablation_external.csv", index=False)

print("\nSorted by external average precision:")
print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results/feature_group_ablation_external.csv")
