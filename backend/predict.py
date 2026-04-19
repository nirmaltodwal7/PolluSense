"""
predict.py
----------
Prediction and forecasting logic for the PolluSense AQI system.

Provides:
  - predict_single(city, date_str) → single AQI prediction
  - forecast_city(city, days)       → list of 1–7 AQI predictions
  - get_forecast_result(city, days) → full structured result dict

The hybrid ensemble averages:
  RF Baseline (40%) + LSTM Prediction (60%) → Final AQI
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from model import (
    DATA_PATH, MODELS_DIR,
    ALL_FEATURE_COLS, SEQ_LEN,
    get_aqi_category, get_health_alert,
    validate_city, validate_days,
    load_rf_model, load_lstm_model,
    load_scaler, load_selected_features
)

# Ensemble weights
RF_WEIGHT   = 0.40
LSTM_WEIGHT = 0.60


# ─── Model Cache (loaded once per process) ────────────────────────────────────
_rf_model        = None
_lstm_model      = None
_scaler          = None
_selected_feats  = None
_aqi_scaler_params = None
_city_data_cache = {}


def _ensure_models_loaded():
    """Lazy-load all models into module-level cache (thread-safe for Flask)."""
    global _rf_model, _lstm_model, _scaler, _selected_feats, _aqi_scaler_params

    if _rf_model is None:
        print("Loading Random Forest model...")
        _rf_model = load_rf_model()

    if _scaler is None:
        _scaler = load_scaler()

    if _selected_feats is None:
        _selected_feats = load_selected_features()

    if _lstm_model is None:
        print("Loading LSTM model (this may take a few seconds)...")
        _lstm_model = load_lstm_model()

    if _aqi_scaler_params is None:
        params_path = os.path.join(MODELS_DIR, "aqi_scaler_params.pkl")
        _aqi_scaler_params = joblib.load(params_path)


def _get_city_history(city: str) -> pd.DataFrame:
    """
    Return cleaned, sorted DataFrame for a single city.
    Cached per-city to avoid repeated I/O.
    """
    if city not in _city_data_cache:
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
        df = df[df["city"] == city].sort_values("date").reset_index(drop=True)

        # Clean missing values
        for col in ALL_FEATURE_COLS + ["AQI"]:
            df[col] = df[col].ffill().fillna(df[col].median())

        _city_data_cache[city] = df

    return _city_data_cache[city].copy()


def _scale_features(feature_values: np.ndarray) -> np.ndarray:
    """Apply the fitted MinMaxScaler to a feature array."""
    return _scaler.transform(feature_values)


def _aqi_to_scaled(aqi: float) -> float:
    """Normalize a raw AQI value using the stored AQI min/max."""
    mn = _aqi_scaler_params["min"]
    mx = _aqi_scaler_params["max"]
    return (aqi - mn) / (mx - mn)


def _scaled_to_aqi(scaled: float) -> float:
    """Inverse-normalize a scaled AQI value back to raw AQI."""
    mn = _aqi_scaler_params["min"]
    mx = _aqi_scaler_params["max"]
    return scaled * (mx - mn) + mn


def _rf_predict(feature_row: np.ndarray) -> float:
    """
    Use Random Forest to predict AQI from a raw feature row.
    feature_row: 1D array of [PM2.5, PM10, NO2, CO, SO2, O3]
    """
    scaled_row = _scaler.transform(feature_row.reshape(1, -1))
    return float(_rf_model.predict(scaled_row)[0])


def _lstm_predict(sequence: np.ndarray) -> float:
    """
    Use LSTM to predict next AQI.
    sequence: array of shape [SEQ_LEN x n_selected_features] (already scaled)
    Returns raw AQI value.
    """
    seq_input = sequence.reshape(1, SEQ_LEN, len(_selected_feats))
    scaled_pred = float(_lstm_model.predict(seq_input, verbose=0)[0, 0])
    return _scaled_to_aqi(scaled_pred)


def _build_lstm_sequence(city_df: pd.DataFrame, end_idx: int) -> np.ndarray:
    """
    Extract and scale the last SEQ_LEN rows ending at end_idx.
    Returns array [SEQ_LEN x n_selected_features] ready for LSTM.
    """
    window = city_df.iloc[end_idx - SEQ_LEN: end_idx][ALL_FEATURE_COLS].values
    scaled = _scaler.transform(window)
    # Keep only selected features (RF-ranked)
    feat_indices = [ALL_FEATURE_COLS.index(f) for f in _selected_feats]
    return scaled[:, feat_indices]


def predict_single(city: str, date_str: str) -> dict:
    """
    Predict AQI for a specific city on a specific date.

    Args:
        city     : City name (case-insensitive)
        date_str : Date string 'YYYY-MM-DD'

    Returns:
        dict with keys: city, date, aqi, category, alert
    """
    _ensure_models_loaded()
    city = validate_city(city)
    target_date = pd.to_datetime(date_str)

    city_df = _get_city_history(city)

    # Find the latest date in history ≤ target_date
    past = city_df[city_df["date"] <= target_date]
    if len(past) < SEQ_LEN:
        # Fallback: use whatever is available
        past = city_df.tail(SEQ_LEN)

    end_idx = len(past)

    # RF prediction from most recent row of features
    latest_row = city_df.iloc[end_idx - 1][ALL_FEATURE_COLS].values
    rf_aqi = _rf_predict(latest_row)

    # LSTM prediction from the last SEQ_LEN rows
    seq = _build_lstm_sequence(city_df, end_idx)
    lstm_aqi = _lstm_predict(seq)

    # Ensemble
    final_aqi = round(RF_WEIGHT * rf_aqi + LSTM_WEIGHT * lstm_aqi, 1)
    final_aqi = max(0, min(500, final_aqi))

    return {
        "city":     city,
        "date":     date_str,
        "aqi":      final_aqi,
        "category": get_aqi_category(final_aqi),
        "alert":    get_health_alert(final_aqi),
    }


def forecast_city(city: str, days: int) -> list[dict]:
    """
    Iteratively forecast AQI for the next `days` days (1–7 max).

    Each iteration:
      1. RF predicts from latest feature row
      2. LSTM predicts from last SEQ_LEN sequence
      3. Ensemble result used to update the pseudo-feature row for next step

    Args:
        city : City name
        days : Number of forecast days (1–7)

    Returns:
        List of dicts with keys: day, date, aqi, category, alert
    """
    _ensure_models_loaded()
    city = validate_city(city)
    validate_days(days)

    city_df = _get_city_history(city)

    # Working sequence: seed with latest SEQ_LEN rows (scaled)
    seed_scaled = _scaler.transform(city_df.tail(SEQ_LEN)[ALL_FEATURE_COLS].values)
    feat_indices = [ALL_FEATURE_COLS.index(f) for f in _selected_feats]
    # LSTM sequence — only selected features
    lstm_seq = seed_scaled[:, feat_indices].copy()   # [SEQ_LEN x n_feats]

    # RF uses the latest feature row (raw for inverse after scale)
    last_raw_row = city_df.tail(1)[ALL_FEATURE_COLS].values.copy()  # [1 x 6]

    results = []
    today = datetime.today()

    for day in range(1, days + 1):
        forecast_date = (today + timedelta(days=day)).strftime("%Y-%m-%d")

        # RF prediction
        rf_aqi = _rf_predict(last_raw_row[0])

        # LSTM prediction
        seq_input  = lstm_seq.reshape(1, SEQ_LEN, len(_selected_feats))
        scaled_out = float(_lstm_model.predict(seq_input, verbose=0)[0, 0])
        lstm_aqi   = _scaled_to_aqi(scaled_out)

        # Ensemble
        final_aqi = round(RF_WEIGHT * rf_aqi + LSTM_WEIGHT * lstm_aqi, 1)
        final_aqi = max(0, min(500, final_aqi))

        results.append({
            "day":      day,
            "date":     forecast_date,
            "aqi":      final_aqi,
            "category": get_aqi_category(final_aqi),
            "alert":    get_health_alert(final_aqi),
        })

        # ── Update state for next iteration ──────────────────────────────────
        # Estimate next feature row by shifting AQI proportionally
        # (use ratio of predicted to last known AQI to scale pollutants)
        last_known_aqi = float(city_df.tail(1)["AQI"].values[0])
        ratio = final_aqi / max(last_known_aqi, 1)
        # Update raw feature row (clamp to realistic range)
        last_raw_row = np.clip(last_raw_row * ratio, 0, None)

        # Scale updated row
        new_scaled_row = _scaler.transform(last_raw_row)
        new_feat_row   = new_scaled_row[:, feat_indices]  # [1 x n_feats]

        # Slide LSTM window: drop oldest, append newest
        lstm_seq = np.concatenate([lstm_seq[1:], new_feat_row], axis=0)

    return results


def get_forecast_result(city: str, days: int) -> dict:
    """
    High-level function returning a full structured prediction result.
    Used directly by the Flask API.

    Returns:
        {
          "city": str,
          "predictions": [float, ...],
          "categories": [str, ...],
          "dates": [str, ...],
          "alert": str | None,
          "forecast": [{"day":int, "date":str, "aqi":float, "category":str}, ...]
        }
    """
    forecasts = forecast_city(city, days)

    predictions  = [f["aqi"]      for f in forecasts]
    categories   = [f["category"] for f in forecasts]
    dates        = [f["date"]     for f in forecasts]

    # Global alert: triggered if ANY day exceeds threshold
    max_aqi = max(predictions)
    alert = get_health_alert(max_aqi)

    return {
        "city":        city,
        "predictions": predictions,
        "categories":  categories,
        "dates":       dates,
        "alert":       alert,
        "forecast":    forecasts,
    }


# ─── Quick Test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing single prediction for Delhi...")
    result = predict_single("Delhi", "2023-06-15")
    print(result)

    print("\nTesting 7-day forecast for Mumbai...")
    forecast = get_forecast_result("Mumbai", 7)
    for f in forecast["forecast"]:
        print(f"  Day {f['day']} ({f['date']}): AQI={f['aqi']}  [{f['category']}]")
    print(f"Alert: {forecast['alert']}")
