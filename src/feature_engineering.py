import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def engineer_features(df):
    """
    Advanced feature engineering with improved metrics and dynamic lag updating.
    
    Features added:
    - lag_28 (monthly trend)
    - rolling_mean_14 (14-day window)
    - rolling_std_14 (14-day volatility)
    - promo_rolling_7 (promotion impact)
    - day_of_month
    - trend (rolling_mean_7 - rolling_mean_14)
    
    All features use shift() to prevent data leakage.
    """
    df = df.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATE PROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"])

    # ─────────────────────────────────────────────────────────────────────────
    # 2. TIME-BASED FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. LAG FEATURES (with shift to prevent leakage)
    # ─────────────────────────────────────────────────────────────────────────
    grp = df.groupby(["store_nbr", "family"])["sales"]

    df["lag_1"] = grp.shift(1)
    df["lag_7"] = grp.shift(7)
    df["lag_14"] = grp.shift(14)
    df["lag_28"] = grp.shift(28)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ROLLING FEATURES (7-day and 14-day windows)
    # ─────────────────────────────────────────────────────────────────────────
    df["rolling_mean_7"] = grp.shift(1).rolling(7, min_periods=1).mean()
    df["rolling_std_7"] = grp.shift(1).rolling(7, min_periods=1).std()
    
    df["rolling_mean_14"] = grp.shift(1).rolling(14, min_periods=1).mean()
    df["rolling_std_14"] = grp.shift(1).rolling(14, min_periods=1).std()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. TREND FEATURE (difference between short & long-term averages)
    # ─────────────────────────────────────────────────────────────────────────
    df["trend"] = df["rolling_mean_7"] - df["rolling_mean_14"]

    # ─────────────────────────────────────────────────────────────────────────
    # 6. PROMOTION FEATURES
    # ─────────────────────────────────────────────────────────────────────────
    df["promo_lag_1"] = df.groupby(["store_nbr", "family"])["onpromotion"].shift(1)
    df["promo_rolling_7"] = (
        df.groupby(["store_nbr", "family"])["onpromotion"]
        .shift(1)
        .rolling(7, min_periods=1)
        .mean()
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. ENCODING CATEGORICAL VARIABLES
    # ─────────────────────────────────────────────────────────────────────────
    for col in ["family", "city", "state", "type"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col + "_enc"] = LabelEncoder().fit_transform(df[col])

    # ─────────────────────────────────────────────────────────────────────────
    # 8. DROP NaN ROWS (from lag features)
    # ─────────────────────────────────────────────────────────────────────────
    # Drop rows with NaN, particularly from lag_28 (need 28-day history)
    df = df.dropna()

    # ─────────────────────────────────────────────────────────────────────────
    # 9. DEFINE FEATURE COLUMNS
    # ─────────────────────────────────────────────────────────────────────────
    feature_cols = [
        # Time features
        "month", "day", "day_of_week", "day_of_month", "quarter",
        "is_weekend", "is_month_start", "is_month_end",
        # Lag features (1, 7, 14, 28 days)
        "lag_1", "lag_7", "lag_14", "lag_28",
        # Rolling features
        "rolling_mean_7", "rolling_std_7",
        "rolling_mean_14", "rolling_std_14",
        # Trend
        "trend",
        # Promotion features
        "onpromotion", "promo_lag_1", "promo_rolling_7",
        # Store identifier
        "store_nbr",
        # Encoded categorical
        "family_enc",
    ]

    # Only include city, state, type if they exist in data
    for col in ["city_enc", "state_enc", "type_enc"]:
        if col in df.columns:
            feature_cols.append(col)

    X = df[feature_cols]
    y = df["sales"]

    return df, X, y, feature_cols


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/demo_sales.csv")
    df_eng, X, y, features = engineer_features(df)

    print("Features shape:", X.shape)
    print("Sample:")
    print(X.head())
    print("\nFeature columns:", features)
