
import pandas as pd
import numpy as np
import tldextract

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["ClassLabel"]).drop_duplicates("URL").copy()

all_features = [
    "url_length", "has_ip_address", "dot_count", "https_flag",
    "url_entropy", "token_count", "subdomain_count",
    "query_param_count", "tld_length", "path_length",
    "has_hyphen_in_domain", "number_of_digits",
    "tld_popularity", "suspicious_file_extension",
    "domain_name_length", "percentage_numeric_chars"
]

reduced_features = [
    f for f in all_features
    if f not in {
        "https_flag", "path_length", "token_count",
        "number_of_digits", "percentage_numeric_chars"
    }
]

extractor = tldextract.TLDExtract(suffix_list_urls=())

def registered_domain(url):
    r = extractor(url)
    if r.domain and r.suffix:
        return r.domain + "." + r.suffix
    return r.domain or "UNPARSED"

df["registered_domain"] = df["URL"].astype(str).apply(registered_domain)
df = df[df["registered_domain"] != "UNPARSED"].copy()

y = df["ClassLabel"].astype(int)
groups = df["registered_domain"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(
    splitter.split(df, y, groups)
)

feature_sets = {
    "Two features": ["https_flag", "has_ip_address"],
    "Eleven features": reduced_features
}

for name, features in feature_sets.items():
    X = df[features].replace([np.inf, -np.inf], np.nan)

    train_valid = X.iloc[train_idx].notna().all(axis=1)
    test_valid = X.iloc[test_idx].notna().all(axis=1)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X.iloc[train_idx].loc[train_valid],
        y.iloc[train_idx].loc[train_valid]
    )

    pred = model.predict(X.iloc[test_idx].loc[test_valid])
    prob = model.predict_proba(X.iloc[test_idx].loc[test_valid])[:, 1]
    y_test = y.iloc[test_idx].loc[test_valid]

    print(f"\n{name}")
    print("Features:", len(features))
    print("Test rows:", len(y_test))
    print("Accuracy:", round(accuracy_score(y_test, pred), 6))
    print("ROC-AUC:", round(roc_auc_score(y_test, prob), 6))
