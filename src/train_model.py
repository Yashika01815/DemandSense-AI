import pandas as pd
import numpy as np
import pickle
import os
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_features


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


def calculate_smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error.
    More robust than MAPE for small values and zero sales.
    
    Formula: SMAPE = 100/n * Σ(2*|y_true - y_pred| / (|y_true| + |y_pred|))
    """
    denominator = np.abs(y_true) + np.abs(y_pred)
    diff = 2.0 * np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # Handle division by zero
    return 100.0 * np.mean(diff)


def calculate_wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error.
    Better for imbalanced data where some records are more important.
    
    Formula: WAPE = Σ|y_true - y_pred| / Σ|y_true|
    """
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-9) * 100


def train(data_path: str):
    """Train XGBoost model with enhanced evaluation metrics."""
    
    # ──────────────────────────────────────────────────────────────────────
    # 1. LOAD DATA
    # ──────────────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Raw shape: {df.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING
    # ──────────────────────────────────────────────────────────────────────
    print("Engineering features...")
    df_eng, X, y, feature_cols = engineer_features(df)
    print(f"  Engineered shape: {X.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # 3. TRAIN/TEST SPLIT (time-aware, no shuffle)
    # ──────────────────────────────────────────────────────────────────────
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")

    # ──────────────────────────────────────────────────────────────────────
    # 4. TRAIN XGBOOST
    # ──────────────────────────────────────────────────────────────────────
    print("Training XGBoost model...")
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ──────────────────────────────────────────────────────────────────────
    # 5. EVALUATE WITH ROBUST METRICS
    # ──────────────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # No negative sales

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Use SMAPE instead of MAPE (numerically stable)
    smape = calculate_smape(y_test.values, y_pred)
    
    # Optional: also calculate WAPE
    wape = calculate_wape(y_test.values, y_pred)

    print("\n── Evaluation Metrics ────────────────────────────")
    print(f"  MAE   : {mae:.3f}")
    print(f"  RMSE  : {rmse:.3f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  SMAPE : {smape:.2f}%")
    print(f"  WAPE  : {wape:.2f}%")
    print("──────────────────────────────────────────────────")

    # ──────────────────────────────────────────────────────────────────────
    # 6. FEATURE IMPORTANCE (TOP 15)
    # ──────────────────────────────────────────────────────────────────────
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False)
    print("\n── Top 15 Feature Importances ──────────────────")
    print(importance.head(15).to_string())
    print("──────────────────────────────────────────────────")

    # ──────────────────────────────────────────────────────────────────────
    # 7. SAVE MODEL
    # ──────────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved → {MODEL_PATH}")

    # ──────────────────────────────────────────────────────────────────────
    # 8. SAVE METADATA
    # ──────────────────────────────────────────────────────────────────────
    meta = {
        "feature_cols": feature_cols,
        "metrics": {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 4),
            "SMAPE": round(smape, 2),
            "WAPE": round(wape, 2),
        },
        "feature_importance": importance.head(15).to_dict(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "data_path": data_path,
        "model_description": (
            "XGBoost regression model trained on time-series sales data. "
            "Features include lag features (1, 7, 14, 28 days), rolling "
            "statistics (7 and 14-day windows), trend indicators, and promotion effects."
        )
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved → {META_PATH}")

    return model, meta


def load_model():
    """Load saved model + metadata. Call from app.py."""
    if not os.path.exists(MODEL_PATH):
        return None, {}
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "demo_sales.csv"
    )
    train(DATA_PATH)
