
import pandas as pd
import numpy as np
import tldextract

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df = pd.read_csv("data/url_features_extracted1.csv")
df = df.dropna(subset=["ClassLabel"]).drop_duplicates("URL").copy()

features = [
    "url_length", "has_ip_address", "dot_count", "https_flag",
    "url_entropy", "token_count", "subdomain_count",
    "query_param_count", "tld_length", "path_length",
    "has_hyphen_in_domain", "number_of_digits",
    "tld_popularity", "suspicious_file_extension",
    "domain_name_length", "percentage_numeric_chars"
]

extractor = tldextract.TLDExtract(suffix_list_urls=())

def registered_domain(url):
    result = extractor(url)
    if result.domain and result.suffix:
        return result.domain + "." + result.suffix
    return result.domain or "UNPARSED"

df["registered_domain"] = df["URL"].astype(str).apply(registered_domain)
df = df[df["registered_domain"] != "UNPARSED"].copy()

X = df[features].replace([np.inf, -np.inf], np.nan)
y = df["ClassLabel"].astype(int)
groups = df["registered_domain"]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

excluded = [
    "https_flag", "path_length", "token_count",
    "number_of_digits", "percentage_numeric_chars"
]

for name, selected in [
    ("Model A: all features", features),
    ("Model B: reduced features", [f for f in features if f not in excluded])
]:
    X_train = X.iloc[train_idx][selected]
    X_test = X.iloc[test_idx][selected]

    train_valid = X_train.notna().all(axis=1)
    test_valid = X_test.notna().all(axis=1)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train.loc[train_valid], y.iloc[train_idx].loc[train_valid])

    predictions = model.predict(X_test.loc[test_valid])
    probabilities = model.predict_proba(X_test.loc[test_valid])[:, 1]
    y_test = y.iloc[test_idx].loc[test_valid]

    print(f"\n{name}")
    print("Features:", len(selected))
    print("Test rows:", len(y_test))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 6))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 6))
    print(classification_report(y_test, predictions))
