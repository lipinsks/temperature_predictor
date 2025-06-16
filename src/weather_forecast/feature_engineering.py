# src/weather_forecast/feature_engineering.py

import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from typing import Optional
from weather_forecast.data_ingest import SQLiteSource, DataSourceError

logger = logging.getLogger(__name__)

class FeatureEngineeringError(Exception):
    """Raised on failures during feature engineering."""
    pass

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop duplicate 'id'
    - Ensure timestamp is datetime
    - Interpolate temperature over time
    - Drop any remaining rows missing core fields
    """
    df = df.copy()
    df = df.drop_duplicates(subset="id")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").set_index("timestamp")
    try:
        df["temperature"] = df["temperature"].interpolate(method="time")
    except Exception as e:
        raise FeatureEngineeringError(f"Time interpolation failed: {e}")
    return df.reset_index().dropna(subset=["temperature","lat","lon","timestamp"])

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical and raw time features:
    year, month, day, hour, dayofyear, weekday, sin/cos transforms.
    """
    df = df.copy()
    ts = df["timestamp"]
    df["year"]      = ts.dt.year
    df["month"]     = ts.dt.month
    df["day"]       = ts.dt.day
    df["hour"]      = ts.dt.hour
    df["dayofyear"] = ts.dt.dayofyear
    df["weekday"]   = ts.dt.weekday
    df["sin_day"]   = np.sin(2*np.pi * df["dayofyear"] / 365.25)
    df["cos_day"]   = np.cos(2*np.pi * df["dayofyear"] / 365.25)
    df["sin_hour"]  = np.sin(2*np.pi * df["hour"] / 24)
    df["cos_hour"]  = np.cos(2*np.pi * df["hour"] / 24)
    return df

def add_api_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject raw and derived features from OpenWeather:
      - RAW: feels_like, temp_min, temp_max, pressure, humidity,
             visibility, wind_speed, wind_deg, wind_gust,
             clouds_all, rain_1h, snow_1h
      - DERIVED:
         * temp_range = temp_max - temp_min
         * temp_diff_feels = feels_like - temperature
         * sin/cos wind direction from wind_deg
         * one-hot encode weather_main
    """
    df = df.copy()

    # Derived temperature metrics
    if {"temp_min","temp_max"}.issubset(df.columns):
        df["temp_range"] = df["temp_max"] - df["temp_min"]
    if {"temperature","feels_like"}.issubset(df.columns):
        df["temp_diff_feels"] = df["feels_like"] - df["temperature"]

    # Wind direction encoding
    if "wind_deg" in df:
        df["sin_wind_dir"] = np.sin(2*np.pi * df["wind_deg"] / 360)
        df["cos_wind_dir"] = np.cos(2*np.pi * df["wind_deg"] / 360)

    # One-hot encode the main weather category
    if "weather_main" in df:
        weather_dummies = pd.get_dummies(df["weather_main"], prefix="weather")
        df = pd.concat([df, weather_dummies], axis=1)

    return df

def add_spatial_features(
    df: pd.DataFrame,
    radius_km: float = 10,
    min_samples: int = 3
) -> pd.DataFrame:
    """
    For each point, compute the mean temperature of neighbors
    within radius_km (using Haversine metric on lat/lon).
    """
    df = df.copy().reset_index(drop=True)
    coords = np.deg2rad(df[["lat","lon"]].values)
    tree   = BallTree(coords, metric="haversine")
    radius = radius_km / 6371.0  # Earth radius in km
    means  = []
    for point in coords:
        idx   = tree.query_radius(point.reshape(1,-1), r=radius)[0]
        temps = df.loc[idx, "temperature"].values
        means.append(np.mean(temps) if len(temps) >= min_samples else np.nan)
    df["temp_mean_neighbors"] = means
    return df

def generate_features_for_inference(
    lat: float,
    lon: float,
    timestamp: pd.Timestamp,
    use_spatial: bool = False,
    spatial_radius_km: float = 10,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Build exactly the same set of features for a single (lat, lon, timestamp).
    Placeholder temperature/id so schema validation passes, then:
      1) time features
      2) optional spatial based on history
      3) drop raw temperature/id
    """
    df = pd.DataFrame({
        "timestamp": [timestamp],
        "lat":       [lat],
        "lon":       [lon],
        "temperature":[np.nan],  # dummy
        "id":         [0],       # dummy
    })
    df = add_time_features(df)
    df = add_api_features(df)  # no-op if raw API cols missing
    if use_spatial:
        if not db_path:
            raise FeatureEngineeringError("db_path required for spatial features")
        df_hist = SQLiteSource(db_path=db_path, table="weather").load()
        all_df  = pd.concat([df_hist[["temperature","lat","lon"]], df], ignore_index=True)
        all_df  = add_spatial_features(all_df, radius_km=spatial_radius_km)
        df["temp_mean_neighbors"] = all_df["temp_mean_neighbors"].iloc[-1]
    return df.set_index("timestamp").drop(columns=["temperature","id"], errors="ignore")

class FeaturePipeline:
    def __init__(self, use_time: bool = True, use_spatial: bool = False, spatial_radius_km: float = 10):
        self.use_time      = use_time
        self.use_spatial   = use_spatial
        self.spatial_radius_km = spatial_radius_km

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Batch feature pipeline for training:
          1) clean
          2) inject API features
          3) time features (if enabled)
          4) spatial features (if enabled)
        """
        try:
            df_clean = clean_data(df)
            df_clean = add_api_features(df_clean)
            if self.use_time:
                df_clean = add_time_features(df_clean)
            if self.use_spatial:
                df_clean = add_spatial_features(df_clean, radius_km=self.spatial_radius_km)
            return df_clean
        except Exception as e:
            raise FeatureEngineeringError(f"Feature pipeline failed: {e}")
