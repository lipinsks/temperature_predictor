# src/weather_forecast/modeling.py

import os
import pickle
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from weather_forecast.data_ingest import SQLiteSource, DataSourceError
from weather_forecast.feature_engineering import (
    FeaturePipeline,
    generate_features_for_inference
)

# Konfiguracja loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s'
)

# Ścieżki domyślne
default_db = Path(__file__).resolve().parents[2] / 'data' / 'weather.db'
model_dir   = Path(__file__).resolve().parents[2] / 'models'
# Nie nadpisujemy już lgbm_full.pkl – jeśli spatial=True, zapisujemy lgbm_spatial.pkl
default_model_name_spatial   = 'lgbm_spatial.pkl'
default_model_name_nospatial = 'lgbm_nospatial.pkl'


def train_full(
    db_path: Path = default_db,
    use_spatial: bool = True
) -> Path:
    """
    Trenuje LightGBM na wszystkich rekordach z bazy.
    Używa cech czasowych i (opcjonalnie) przestrzennych.
    Zapisuje wytrenowany model do pliku (w zależności od use_spatial).
    Zwraca ścieżkę do pliku z modelem.
    """
    # 1) Wczytaj dane
    try:
        src = SQLiteSource(db_path=str(db_path), table='weather')
        df_raw = src.load()
        logging.info(f"Loaded {len(df_raw):,} records from {db_path}")
    except DataSourceError as e:
        logging.error("Data ingest failed: %s", e)
        return None

    # 2) Featuryzacja
    pipeline = FeaturePipeline(
        use_time=True,
        use_spatial=use_spatial,
        spatial_radius_km=10
    )
    df_feat = pipeline.transform(df_raw)
    df_feat = df_feat.set_index('timestamp')
    logging.info(
        f"After feature engineering: {df_feat.shape[0]:,} rows, {df_feat.shape[1]} features"
    )

    # 3) Przygotuj X, y
    X = df_feat.drop(columns=['temperature', 'city', 'country', 'id'], errors='ignore')
    y = df_feat['temperature']
    logging.info("Prepared X shape %s, y length %d", X.shape, len(y))

    # 4) Podziel na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    logging.info("Split train/test: %s / %s", X_train.shape, X_test.shape)

    # 5) Trenowanie
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8
    )
    logging.info("Starting training on %d samples...", len(X_train))
    model.fit(X_train, y_train)
    logging.info("Training completed")

    # 6) Ewaluacja
    preds = model.predict(X_test)
    rmse = ((preds - y_test) ** 2).mean() ** 0.5
    logging.info(f"Test RMSE: {rmse:.3f}")

    # 7) Zapis modelu pod odpowiednią nazwą
    model_dir.mkdir(parents=True, exist_ok=True)
    model_filename = (
        default_model_name_spatial   if use_spatial
        else default_model_name_nospatial
    )
    model_path = model_dir / model_filename
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'use_spatial': use_spatial}, f)
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
    Wczytuje zapisany model i generuje predykcję temperatury
    dla pojedynczego punktu (lat, lon) i czasu.
    """
    # Autodetekcja ścieżki, jeśli nie podano:
    if model_path is None:
        # zakładamy, że chcesz spatial
        model_path = model_dir / default_model_name_spatial

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    use_spatial = data.get('use_spatial', False)

    # 1) Timestamp
    if timestamp:
        ts = pd.to_datetime(timestamp)
    else:
        df_last = SQLiteSource(db_path=str(db_path), table='weather').load()
        ts = df_last['timestamp'].max()

    # 2) Generacja cech dla próbki
    df_feat = generate_features_for_inference(
        lat=lat,
        lon=lon,
        timestamp=ts,
        use_spatial=use_spatial,
        spatial_radius_km=10,
        db_path=str(db_path)
    )

    # 3) Przygotuj X
    X = df_feat.drop(
        columns=[col for col in ['temperature', 'city', 'country', 'id'] if col in df_feat],
        errors='ignore'
    )

    # 4) Predykcja
    pred = model.predict(X)
    return pd.Series(pred, index=X.index, name='pred_temperature')


if __name__ == '__main__':
    # Teraz jawnie trenujemy z spatial=True (i zapisujemy lgbm_spatial.pkl)
    path = train_full(use_spatial=False)
    if path is None:
        logging.error("Training failed.")
        exit(1)

    # Demo-inference możesz odkomentować, jeśli chcesz (to będzie szybkie tylko jeśli use_spatial=False):
    # result = inference(lat=50.0, lon=19.9, timestamp="2025-06-11T15:00:00", model_path=path)
    # print("Demo inference result:")
    # print(result.to_frame())