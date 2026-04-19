"""
train_lstm.py
-------------
Trains an LSTM neural network using the top features selected by
the Random Forest model.

Architecture:
    Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(1)

Steps:
  1. Load dataset and apply same cleaning as RF training
  2. Use RF-selected top features only
  3. Build sliding-window sequences (SEQ_LEN days)
  4. Train Keras LSTM model
  5. Save model to models/aqi_lstm_model.h5

Run:
    python train_lstm.py  (run AFTER train_random_forest.py)
"""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error

from model import (
    DATA_PATH, MODELS_DIR, LSTM_MODEL_PATH,
    SCALER_PATH, FEATURES_PATH,
    ALL_FEATURE_COLS, SEQ_LEN
)

# ─── Hyperparameters ──────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 32
VALIDATION_SPLIT = 0.15
LSTM_UNITS_1     = 128
LSTM_UNITS_2     = 64
DROPOUT_RATE     = 0.2


def load_and_clean(path: str) -> pd.DataFrame:
    """Load and clean the dataset (same logic as RF training)."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df[ALL_FEATURE_COLS + ["AQI"]] = (
        df.groupby("city")[ALL_FEATURE_COLS + ["AQI"]]
        .transform(lambda x: x.ffill())
    )
    for col in ALL_FEATURE_COLS + ["AQI"]:
        df[col] = df[col].fillna(df[col].median())

    return df


def build_sequences(city_data: np.ndarray, aqi_col_idx: int, seq_len: int):
    """
    Build sliding-window X/y pairs for LSTM training.

    city_data : 2D array [timesteps x n_features+1]  (features + AQI)
    aqi_col_idx : index of AQI column in city_data
    seq_len : look-back window size

    Returns X [samples x seq_len x n_features], y [samples]
    """
    X_list, y_list = [], []
    for i in range(seq_len, len(city_data)):
        X_list.append(city_data[i - seq_len: i, :-1])   # features only
        y_list.append(city_data[i, aqi_col_idx])          # target: AQI (scaled)
    return np.array(X_list), np.array(y_list)


def build_model(input_shape: tuple) -> keras.Model:
    """Construct and compile the LSTM model."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(LSTM_UNITS_1, return_sequences=True,
                    name="lstm_1"),
        layers.Dropout(DROPOUT_RATE, name="dropout_1"),
        layers.LSTM(LSTM_UNITS_2, return_sequences=False,
                    name="lstm_2"),
        layers.Dropout(DROPOUT_RATE, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_hidden"),
        layers.Dense(1, name="output"),
    ], name="AQI_LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


def train():
    """Full LSTM training pipeline."""
    print("Loading cleaned dataset...")
    df = load_and_clean(DATA_PATH)

    # Load scaler and selected features from RF step
    scaler: object           = joblib.load(SCALER_PATH)
    selected_features: list  = joblib.load(FEATURES_PATH)
    print(f"Using features: {selected_features}")

    # Scale all feature columns (same scaler as RF for consistency)
    df[ALL_FEATURE_COLS] = scaler.transform(df[ALL_FEATURE_COLS])

    # Also scale AQI independently (0–500 range → 0–1)
    aqi_max = df["AQI"].max()
    aqi_min = df["AQI"].min()
    df["AQI_scaled"] = (df["AQI"] - aqi_min) / (aqi_max - aqi_min)

    # Save AQI scaler params for inverse transform at prediction time
    aqi_scaler_path = os.path.join(MODELS_DIR, "aqi_scaler_params.pkl")
    joblib.dump({"min": aqi_min, "max": aqi_max}, aqi_scaler_path)
    print(f"AQI range: {aqi_min:.1f} – {aqi_max:.1f}")

    # Build sequences per city and combine
    all_X, all_y = [], []
    cities = df["city"].unique()

    for city in cities:
        city_df = df[df["city"] == city].sort_values("date")
        # Combine selected features + scaled AQI as last column for target
        cols = selected_features + ["AQI_scaled"]
        city_array = city_df[cols].values
        X, y = build_sequences(city_array, aqi_col_idx=-1, seq_len=SEQ_LEN)
        all_X.append(X)
        all_y.append(y)
        print(f"  {city}: {len(X)} sequences")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    print(f"\nTotal sequences: {len(X_all):,}  |  Input shape: {X_all.shape}")

    # Shuffle for training
    idx = np.random.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    # ── Build & Train ────────────────────────────────────────────────────────
    model = build_model(input_shape=(SEQ_LEN, len(selected_features)))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]

    print(f"\nTraining LSTM for up to {EPOCHS} epochs...")
    history = model.fit(
        X_all, y_all,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    split = int(len(X_all) * (1 - VALIDATION_SPLIT))
    y_pred_scaled = model.predict(X_all[split:], verbose=0).flatten()
    y_true_scaled = y_all[split:]

    # Inverse-scale for readable metrics
    y_pred = y_pred_scaled * (aqi_max - aqi_min) + aqi_min
    y_true = y_true_scaled * (aqi_max - aqi_min) + aqi_min
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\nLSTM Validation MAE (original scale): {mae:.2f} AQI units")

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(LSTM_MODEL_PATH)
    print(f"\n✅ LSTM model saved → {LSTM_MODEL_PATH}")
    print("LSTM training complete!")


if __name__ == "__main__":
    train()
