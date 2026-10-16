import pandas as pd

path = "data/url_features_extracted1.csv"
df = pd.read_csv(path)

print("Columns:", df.columns.tolist())
print("\nClassLabel counts, including missing:")
print(df["ClassLabel"].value_counts(dropna=False).sort_index())

print("\nExample URLs grouped by ClassLabel:")
for label, group in df.dropna(subset=["ClassLabel"]).groupby("ClassLabel"):
    print(f"\nClassLabel = {label}")
    print(group["URL"].head(10).to_string(index=False))

print("\nDataset metadata:")
print("Rows:", len(df))
print("Missing labels:", df["ClassLabel"].isna().sum())
