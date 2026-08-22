from analysis.data_loader import load_dataset

df = load_dataset()
print("Dataset loaded successfully.")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nNumber of unique nodes:")
print(df["node_id"].nunique())

print("\nNodes:")
print(df["node_id"].unique())

print("\nRecords per node:")
print(df["node_id"].value_counts())

print("\nStatistical summary:")
print(df.describe())

