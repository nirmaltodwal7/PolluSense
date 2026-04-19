"""
generate_dataset.py
-------------------
Generates a synthetic air quality dataset for demonstration purposes.
Produces 3 years of daily AQI data for 10 major Indian cities with
realistic seasonal patterns and pollutant correlations.

Run:
    python generate_dataset.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─── Configuration ────────────────────────────────────────────────────────────
CITIES = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

# Base AQI levels per city (higher = more polluted baseline)
CITY_BASE_AQI = {
    "Delhi":     180,
    "Lucknow":   160,
    "Kolkata":   145,
    "Ahmedabad": 130,
    "Jaipur":    125,
    "Hyderabad": 110,
    "Mumbai":    105,
    "Pune":       90,
    "Chennai":    85,
    "Bangalore":  75,
}

START_DATE = datetime(2021, 1, 1)
END_DATE   = datetime(2023, 12, 31)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "air_quality_data.csv")

np.random.seed(42)


def seasonal_factor(date: datetime) -> float:
    """Returns a multiplier based on month — winter months are worse."""
    month = date.month
    # Winter (Nov–Feb) → high pollution; Summer → lower
    seasonal = {
        1: 1.40, 2: 1.30, 3: 1.05, 4: 0.90,
        5: 0.85, 6: 0.80, 7: 0.75, 8: 0.78,
        9: 0.88, 10: 1.10, 11: 1.30, 12: 1.45
    }
    return seasonal[month]


def generate_city_data(city: str) -> pd.DataFrame:
    """Generate daily AQI & pollutant readings for one city."""
    dates     = []
    aqi_vals  = []
    pm25_vals = []
    pm10_vals = []
    no2_vals  = []
    co_vals   = []
    so2_vals  = []
    o3_vals   = []

    base_aqi = CITY_BASE_AQI[city]
    current_date = START_DATE

    # Running AQI with momentum (realistic day-to-day continuity)
    current_aqi = base_aqi + np.random.uniform(-20, 20)

    while current_date <= END_DATE:
        sf     = seasonal_factor(current_date)
        target = base_aqi * sf

        # Drift current AQI toward target with noise
        current_aqi += 0.3 * (target - current_aqi) + np.random.normal(0, 12)
        current_aqi  = max(10, min(500, current_aqi))  # clamp

        aqi = round(current_aqi, 1)

        # Derive pollutants from AQI with realistic ratios + individual noise
        pm25 = round(aqi * 0.45 + np.random.normal(0, 5), 2)
        pm10 = round(aqi * 0.70 + np.random.normal(0, 8), 2)
        no2  = round(aqi * 0.25 + np.random.normal(0, 6), 2)
        co   = round(aqi * 0.008 + np.random.normal(0, 0.3), 3)
        so2  = round(aqi * 0.12 + np.random.normal(0, 4), 2)
        o3   = round(aqi * 0.15 + np.random.normal(0, 5), 2)

        # Clamp all pollutants to non-negative values
        pm25 = max(0, pm25)
        pm10 = max(0, pm10)
        no2  = max(0, no2)
        co   = max(0, co)
        so2  = max(0, so2)
        o3   = max(0, o3)

        dates.append(current_date.strftime("%Y-%m-%d"))
        aqi_vals.append(aqi)
        pm25_vals.append(pm25)
        pm10_vals.append(pm10)
        no2_vals.append(no2)
        co_vals.append(co)
        so2_vals.append(so2)
        o3_vals.append(o3)

        current_date += timedelta(days=1)

    df = pd.DataFrame({
        "date":   dates,
        "city":   city,
        "AQI":    aqi_vals,
        "PM2.5":  pm25_vals,
        "PM10":   pm10_vals,
        "NO2":    no2_vals,
        "CO":     co_vals,
        "SO2":    so2_vals,
        "O3":     o3_vals,
    })

    # Randomly introduce ~2% missing values (for realism)
    for col in ["PM2.5", "PM10", "NO2", "CO", "SO2", "O3"]:
        mask = np.random.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating synthetic air quality dataset for {len(CITIES)} cities...")

    all_dfs = []
    for city in CITIES:
        print(f"  → {city}...")
        city_df = generate_city_data(city)
        all_dfs.append(city_df)

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Dataset saved to: {OUTPUT_CSV}")
    print(f"   Rows: {len(df):,}  |  Cities: {df['city'].nunique()}  |  "
          f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\nAQI Statistics:")
    print(df.groupby("city")["AQI"].agg(["mean", "min", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
