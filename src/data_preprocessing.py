import pandas as pd

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("Loading datasets...")
train = pd.read_csv("train.csv")
stores = pd.read_csv("stores.csv")

# -----------------------------
# 2. BASIC CLEANING
# -----------------------------
print("Cleaning data...")

# Convert date
train["date"] = pd.to_datetime(train["date"])

# Remove negative or null sales (important)
train = train[train["sales"] >= 0]
train = train.dropna()

# -----------------------------
# 3. MERGE DATASETS
# -----------------------------
print("Merging datasets...")

df = pd.merge(train, stores, on="store_nbr", how="left")

# -----------------------------
# 4. SELECT IMPORTANT COLUMNS
# -----------------------------
df = df[[
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion",
    "city",
    "state",
    "type"
]]

# -----------------------------
# 5. REDUCE DATA SIZE (SMART TRIM)
# -----------------------------
print("Trimming dataset...")

# Select top stores
top_stores = df["store_nbr"].unique()[:5]

# Select top product categories
top_families = df["family"].unique()[:3]

df = df[
    (df["store_nbr"].isin(top_stores)) &
    (df["family"].isin(top_families))
]

# Keep recent data (last 6 months approx)
df = df.sort_values("date")
df = df.tail(2000)

# -----------------------------
# 6. FEATURE ENGINEERING
# -----------------------------
print("Adding features...")

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek

# Optional: weekend flag
df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)

# -----------------------------
# 7. FINAL CHECK
# -----------------------------
print("\nFinal dataset shape:", df.shape)
print(df.head())

# -----------------------------
# 8. SAVE FILES (FOR YOUR DEMO)
# -----------------------------
print("Saving files...")

df.to_csv("demo_sales.csv", index=False)
df.to_excel("demo_sales.xlsx", index=False)

print("✅ Done! Files created:")
print("- demo_sales.csv")
print("- demo_sales.xlsx")