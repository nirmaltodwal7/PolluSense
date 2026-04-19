"""
model.py
--------
Shared model utilities, constants, and AQI helper functions.
Imported by train_*.py, predict.py, and app.py.
"""

import os
import joblib
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_PATH   = os.path.join(BASE_DIR, "data", "air_quality_data.csv")

RF_MODEL_PATH  = os.path.join(MODELS_DIR, "random_forest_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "aqi_lstm_model.h5")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH  = os.path.join(MODELS_DIR, "selected_features.pkl")

# ─── Constants ────────────────────────────────────────────────────────────────
# All possible input feature columns
ALL_FEATURE_COLS = ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]

# Number of past days used as LSTM input window
SEQ_LEN = 60

# Supported cities
SUPPORTED_CITIES = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

# AQI warning threshold
ALERT_THRESHOLD = 150


# ─── AQI Category & Health Alert ──────────────────────────────────────────────

def get_aqi_category(aqi: float) -> str:
    """Map a numeric AQI value to its WHO/CPCB category string."""
    aqi = float(aqi)
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_health_alert(aqi: float) -> str | None:
    """
    Return a health warning message if AQI exceeds the alert threshold,
    otherwise return None.
    """
    aqi = float(aqi)
    if aqi > 300:
        return (
            "⚠️ HAZARDOUS: Emergency conditions! Everyone is affected. "
            "Avoid all outdoor activities. Wear N95 masks indoors."
        )
    elif aqi > 200:
        return (
            "⚠️ VERY UNHEALTHY: Health alert — everyone may experience "
            "serious health effects. Stay indoors with air purifiers."
        )
    elif aqi > ALERT_THRESHOLD:
        return (
            "⚠️ UNHEALTHY: Members of sensitive groups (children, elderly, "
            "those with respiratory conditions) should avoid prolonged outdoor exposure."
        )
    return None


# ─── Model Loaders ────────────────────────────────────────────────────────────

def load_rf_model():
    """Load and return the trained Random Forest model from disk."""
    if not os.path.exists(RF_MODEL_PATH):
        raise FileNotFoundError(
            f"Random Forest model not found at {RF_MODEL_PATH}. "
            "Please run train_random_forest.py first."
        )
    return joblib.load(RF_MODEL_PATH)


def load_scaler():
    """Load and return the fitted MinMaxScaler from disk."""
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Please run train_random_forest.py first."
        )
    return joblib.load(SCALER_PATH)


def load_selected_features():
    """Load the list of RF-selected top features used for LSTM input."""
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Selected features not found at {FEATURES_PATH}. "
            "Please run train_random_forest.py first."
        )
    return joblib.load(FEATURES_PATH)


def load_lstm_model():
    """Load and return the trained Keras LSTM model from disk."""
    # Import here to avoid loading TF on every import
    import tensorflow as tf
    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(
            f"LSTM model not found at {LSTM_MODEL_PATH}. "
            "Please run train_lstm.py first."
        )
    return tf.keras.models.load_model(LSTM_MODEL_PATH)


# ─── Utility ──────────────────────────────────────────────────────────────────

def validate_days(days: int) -> None:
    """Raise ValueError if days is outside the allowed range [1, 7]."""
    if not isinstance(days, int) or days < 1 or days > 7:
        raise ValueError(f"'days' must be an integer between 1 and 7 (got {days}).")


def validate_city(city: str) -> str:
    """
    Validate and normalize city name.
    Returns the properly-cased city name or raises ValueError.
    """
    city_map = {c.lower(): c for c in SUPPORTED_CITIES}
    normalized = city_map.get(city.strip().lower())
    if normalized is None:
        raise ValueError(
            f"City '{city}' is not supported. "
            f"Supported cities: {', '.join(SUPPORTED_CITIES)}"
        )
    return normalized
