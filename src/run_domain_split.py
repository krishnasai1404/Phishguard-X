
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["URL", "ClassLabel"])
df = df.drop_duplicates(subset=["URL"]).copy()

extractor = tldextract.TLDExtract(suffix_list_urls=())
def registered_domain(url):
    parts = extractor(url)
    if not parts.domain or not parts.suffix:
        return parts.domain or parts.domain
    return parts.domain + "." + parts.suffix

df["domain"] = df["URL"].astype(str).map(registered_domain)
df = df[df["domain"] != ""].copy()

excluded = ["URL", "ClassLabel", "https_flag",
            "path_length", "token_count", "number_of_digits",
            "percentage_numeric_chars", "domain"]
cols = [c for c in df.columns if c not in excluded]

X = df[cols].apply(pd.to_numeric, errors="coerce")
y = df["ClassLabel"].astype(int)
valid = X.notna().all(axis=1)
X, y = X.loc[valid], y.loc[valid]
groups = df.loc[valid, "domain"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train, test = next(splitter.split(X, y, groups))

model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
model.fit(X.iloc[train], y.iloc[train])
prob = model.predict_proba(X.iloc[test])[:, 1]
pred = model.predict(X.iloc[test])

print("Train rows:", len(train))
print("Test rows:", len(test))
print("Shared registered domains:",
      len(set(groups.iloc[train]) & set(groups.iloc[test])))
print("Accuracy:", round(accuracy_score(y.iloc[test], pred), 6))
print("ROC-AUC:", round(roc_auc_score(y.iloc[test], prob), 6))
print(classification_report(y.iloc[test], pred, zero_division=0))
