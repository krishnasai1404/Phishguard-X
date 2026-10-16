
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df.drop_duplicates(subset=["URL"]).copy()

df["host"] = df["URL"].map(
    lambda u: (urlparse(
        u if "://" in u else "http://" + u
    ).hostname or "").lower()
)
df = df[df["host"] != ""].copy()

cols = [
    c for c in df.columns
    if c not in ["URL", "ClassLabel", "https_flag", "host"]
]
X = df[cols].apply(pd.to_numeric, errors="coerce")
y = df["ClassLabel"].astype(int)

valid = X.notna().all(axis=1)
X, y = X.loc[valid], y.loc[valid]
groups = df.loc[valid, "host"]

splitter = GroupShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42
)
train, test = next(splitter.split(X, y, groups))

model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X.iloc[train], y.iloc[train])

pred = model.predict(X.iloc[test])
prob = model.predict_proba(X.iloc[test])[:, 1]

print("Features used:", cols)
print("Train rows:", len(train))
print("Test rows:", len(test))
print("Shared hostnames:", len(
    set(groups.iloc[train]) & set(groups.iloc[test])
))
print(classification_report(y.iloc[test], pred, zero_division=0))
print("ROC-AUC:", round(roc_auc_score(y.iloc[test], prob), 6))
