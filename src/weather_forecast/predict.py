#!/usr/bin/env python3
"""
predict.py

Provides a reusable `predict(lat, lon, timestamp)` function for your API
and also a CLI entrypoint when run as a script.

- `predict(lat, lon, timestamp)` returns a float: the predicted temperature in °C.
- `__main__` parses command‐line args and prints the result.
"""

import sys
import logging
import pickle
from pathlib import Path

import requests
import pandas as pd
import numpy as np

from weather_forecast.feature_engineering import add_time_features, add_api_features

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY    = "19bf03cf474f639062b58c74563d2e38"
UNITS      = "metric"
MODEL_PATH = Path(__file__).parents[2] / "models" / "lgbm_nospatial.pkl"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_current_weather(lat: float, lon: float) -> dict:
    """
    Call OpenWeather API and return JSON dict.
    """
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={API_KEY}&units={UNITS}"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def build_feature_frame(lat: float, lon: float, timestamp: str) -> pd.DataFrame:
    """
    1) Fetch current weather JSON for (lat, lon)
    2) Extract all raw fields needed by the model
    3) Build a single‐row DataFrame
    4) Add time features and API‐derived features
    """
    js = fetch_current_weather(lat, lon)

    row = {
        "timestamp":    pd.to_datetime(timestamp),
        "lat":          lat,
        "lon":          lon,
        # raw OpenWeather fields
        "feels_like":   js["main"]["feels_like"],
        "temp_min":     js["main"]["temp_min"],
        "temp_max":     js["main"]["temp_max"],
        "pressure":     js["main"]["pressure"],
        "humidity":     js["main"]["humidity"],
        "sea_level":    js["main"].get("sea_level", np.nan),
        "grnd_level":   js["main"].get("grnd_level", np.nan),
        "visibility":   js.get("visibility", np.nan),
        "wind_speed":   js["wind"]["speed"],
        "wind_deg":     js["wind"]["deg"],
        "wind_gust":    js["wind"].get("gust", np.nan),
        "clouds_all":   js["clouds"]["all"],
        "rain_1h":      js.get("rain", {}).get("1h", 0.0),
        "snow_1h":      js.get("snow", {}).get("1h", 0.0),
        "weather_id":   js["weather"][0]["id"],
        "weather_main": js["weather"][0]["main"],
        "weather_desc": js["weather"][0]["description"],
        "weather_icon": js["weather"][0]["icon"],
        "dt":           js["dt"],
        "sunrise":      js["sys"]["sunrise"],
        "sunset":       js["sys"]["sunset"],
        "timezone":     js["timezone"],
        "city_id":      js["id"],
        "city_name":    js["name"],
        # placeholders so feature‐engineering code runs without error
        "temperature":  np.nan,
        "id":           0,
    }

    df = pd.DataFrame([row])
    df = add_time_features(df)   # year, month, day, hour, sin/cos, etc.
    df = add_api_features(df)    # temp_range, wind sin/cos, one‐hots, etc.
    return df.set_index("timestamp")


# ── Public API ────────────────────────────────────────────────────────────────

def predict(lat: float, lon: float, timestamp: str) -> float:
    """
    Predict temperature (°C) at the given coordinates and ISO‐format timestamp.

    :param lat: latitude
    :param lon: longitude
    :param timestamp: ISO datetime string, e.g. "2025-06-16T16:00:00"
    :return: predicted temperature in degrees Celsius
    """
    logging.info("Building features for (lat=%s, lon=%s) at %s", lat, lon, timestamp)
    df_feat = build_feature_frame(lat, lon, timestamp)

    logging.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        mdl_data = pickle.load(f)
    model = mdl_data["model"]

    # Ensure feature order matches training
    feature_names = model.booster_.feature_name()
    X = df_feat[feature_names]

    logging.info("Running prediction")
    return float(model.predict(X)[0])


# ── CLI Entrypoint ────────────────────────────────────────────────────────────

def _main():
    if len(sys.argv) != 4:
        print("Usage: python predict.py <lat> <lon> <YYYY-MM-DDThh:mm:ss>")
        sys.exit(1)

    try:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
        ts  = sys.argv[3]
    except ValueError:
        print("Error: lat and lon must be numbers.")
        sys.exit(1)

    temp = predict(lat, lon, ts)
    print(f"Predicted temperature: {temp:.2f} °C")


if __name__ == "__main__":
    _main()
