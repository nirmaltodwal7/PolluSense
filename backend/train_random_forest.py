"""
train_random_forest.py
----------------------
Trains a Random Forest Regressor on the air quality dataset.

Steps:
  1. Load and clean the CSV dataset
  2. Handle missing values (forward-fill then median imputation)
  3. Sort by city + date
  4. Scale features with MinMaxScaler
  5. Train RandomForestRegressor
  6. Extract top feature importances
  7. Save model, scaler, and selected features to /models/

Run:
    python train_random_forest.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from model import (
    DATA_PATH, MODELS_DIR, RF_MODEL_PATH,
    SCALER_PATH, FEATURES_PATH,
    ALL_FEATURE_COLS
)

# Number of top features to select for LSTM input
TOP_N_FEATURES = 4


def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, parse dates, clean missing values, sort by city+date."""
    print("Loading dataset...")
    df = pd.read_csv(path, parse_dates=["date"])

    print(f"  Raw rows: {len(df):,}  |  Missing before cleaning:")
    print(df[ALL_FEATURE_COLS + ["AQI"]].isnull().sum().to_string())

    # Sort so that forward-fill works correctly per city
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    # Forward-fill within each city group (propagate last known value)
    df[ALL_FEATURE_COLS + ["AQI"]] = (
        df.groupby("city")[ALL_FEATURE_COLS + ["AQI"]]
        .transform(lambda x: x.ffill())
    )

    # Fill any remaining NaNs (at start of series) with column median
    for col in ALL_FEATURE_COLS + ["AQI"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    print(f"\n  Missing after cleaning: {df[ALL_FEATURE_COLS + ['AQI']].isnull().sum().sum()}")
    return df


def train(df: pd.DataFrame):
    """Fit MinMaxScaler and RandomForest, then save artifacts."""
    X = df[ALL_FEATURE_COLS].values
    y = df["AQI"].values

    # ── Scaling ──────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train / Test Split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=42
    )

    # ── Random Forest ────────────────────────────────────────────────────────
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = rf_model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"  RF Test MAE : {mae:.2f}")
    print(f"  RF Test R²  : {r2:.4f}")

    # ── Feature Importance ───────────────────────────────────────────────────
    importances = pd.Series(rf_model.feature_importances_, index=ALL_FEATURE_COLS)
    importances = importances.sort_values(ascending=False)
    print(f"\nFeature Importances:")
    print(importances.to_string())

    # Select top N features for LSTM
    selected_features = importances.head(TOP_N_FEATURES).index.tolist()
    print(f"\nTop {TOP_N_FEATURES} features selected for LSTM: {selected_features}")

    # ── Save Artifacts ────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf_model, RF_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(selected_features, FEATURES_PATH)

    print(f"\n✅ Random Forest model saved → {RF_MODEL_PATH}")
    print(f"✅ Scaler saved              → {SCALER_PATH}")
    print(f"✅ Selected features saved   → {FEATURES_PATH}")

    return rf_model, scaler, selected_features


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    train(df)
    print("\nRandom Forest training complete!")
