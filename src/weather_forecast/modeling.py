# src/weather_forecast/modeling.py

import os
import pickle
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from weather_forecast.data_ingest import SQLiteSource, DataSourceError
from weather_forecast.feature_engineering import FeaturePipeline, generate_features_for_inference

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

# Default paths
default_db                  = Path(__file__).resolve().parents[2] / "data" / "weather.db"
model_dir                   = Path(__file__).resolve().parents[2] / "models"
default_model_name_spatial  = "lgbm_spatial.pkl"
default_model_name_nospatial= "lgbm_nospatial.pkl"

def train_full(db_path: Path = default_db, use_spatial: bool = True) -> Path:
    """
    1) Load data from SQLite
    2) Build features (API + time + optional spatial)
    3) Split into train/test
    4) Train LightGBM
    5) Evaluate RMSE
    6) Save model to disk
    """
    # --- 1) Data ingest ---
    try:
        src    = SQLiteSource(db_path=str(db_path), table="weather")
        df_raw = src.load()
        logging.info(f"Loaded {len(df_raw):,} rows from {db_path}")
    except DataSourceError as e:
        logging.error("Data ingest failed: %s", e)
        return None

    # --- 2) Feature engineering ---
    pipeline = FeaturePipeline(use_time=True, use_spatial=use_spatial, spatial_radius_km=10)
    df_feat  = pipeline.transform(df_raw).set_index("timestamp")
    logging.info(f"After featurization: {df_feat.shape[0]:,} rows, {df_feat.shape[1]} features")

    # --- 3) Prepare X, y ---
    # Drop:
    # - original target & identifiers
    # - any object/string columns that LightGBM can't handle directly
    X = df_feat.drop(
        columns=[
            "temperature",   # the target
            "city",          # raw text
            "country",       # raw text
            "id",            # identifier
            "weather_main",  # text
            "weather_desc",  # text
            "weather_icon",  # text
            "city_name"      # text
        ],
        errors="ignore"
    )
    y = df_feat["temperature"]
    logging.info(f"Features used for training: {list(X.columns)}")
    logging.info(f"Prepared X shape {X.shape}, y length {len(y)}")

    # --- 4) Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    logging.info(f"Train/test sizes: {X_train.shape} / {X_test.shape}")

    # --- 5) Model training ---
    model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8)
    logging.info(f"Starting training on {len(X_train):,} samples…")
    model.fit(X_train, y_train)
    logging.info("Training completed")

    # --- 6) Evaluation ---
    preds = model.predict(X_test)
    rmse  = ((preds - y_test) ** 2).mean() ** 0.5
    logging.info(f"Test RMSE: {rmse:.3f}")

    # --- 7) Save model ---
    model_dir.mkdir(parents=True, exist_ok=True)
    model_name = default_model_name_spatial if use_spatial else default_model_name_nospatial
    model_path = model_dir / model_name
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "use_spatial": use_spatial}, f)
    logging.info(f"Saved model to {model_path}")

    return model_path

def inference(
    lat: float,
    lon: float,
    timestamp: str = None,
    model_path: Path = None,
    db_path: Path = default_db
) -> pd.Series:
    """
    1) Load the chosen model
    2) Determine timestamp (explicit or latest in DB)
    3) Generate features for this one point
    4) Drop any unsupported columns
    5) Return temperature prediction
    """
    # Load model & settings
    if model_path is None:
        model_path = model_dir / default_model_name_spatial
    data        = pickle.load(open(model_path, "rb"))
    model       = data["model"]
    use_spatial = data.get("use_spatial", False)

    # Timestamp resolution
    if timestamp:
        ts = pd.to_datetime(timestamp)
    else:
        df_last = SQLiteSource(db_path=str(db_path), table="weather").load()
        ts      = df_last["timestamp"].max()

    # Build features for this sample
    df_feat = generate_features_for_inference(
        lat=lat,
        lon=lon,
        timestamp=ts,
        use_spatial=use_spatial,
        spatial_radius_km=10,
        db_path=str(db_path)
    )

    # Drop unsupported text columns
    X = df_feat.drop(
        columns=["temperature","city","country","id","weather_main","weather_desc","weather_icon","city_name"],
        errors="ignore"
    )

    # Predict
    pred = model.predict(X)
    return pd.Series(pred, index=X.index, name="pred_temperature")


if __name__ == "__main__":
    # Example: train a non-spatial model
    path = train_full(use_spatial=False)
    if path is None:
        logging.error("Training failed.")
        exit(1)
    # To run an inference demo, uncomment:
    # print(inference(lat=50.0, lon=19.9, timestamp="2025-06-11T15:00:00", model_path=path))
