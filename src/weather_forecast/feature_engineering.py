import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from typing import Optional, Dict, Any
from weather_forecast.data_ingest import SQLiteSource, DataSourceError

logger = logging.getLogger(__name__)

class FeatureEngineeringError(Exception):
    """Wyjątek zgłaszany przy problemach w feature_engineering."""
    pass

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # usuń duplikaty po 'id'
    df = df.drop_duplicates(subset='id')
    # timestamp → datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='raise')
    # interpolacja temperatury wg czasu
    df = df.sort_values('timestamp').set_index('timestamp')
    try:
        df['temperature'] = df['temperature'].interpolate(method='time')
    except Exception as e:
        raise FeatureEngineeringError(f"Time interpolation failed: {e}")
    df = df.reset_index().dropna(subset=['temperature','lat','lon','timestamp'])
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df['timestamp']
    df['year']       = ts.dt.year
    df['month']      = ts.dt.month
    df['day']        = ts.dt.day
    df['hour']       = ts.dt.hour
    df['dayofyear']  = ts.dt.dayofyear
    df['weekday']    = ts.dt.weekday
    df['sin_day']    = np.sin(2*np.pi*df['dayofyear']/365.25)
    df['cos_day']    = np.cos(2*np.pi*df['dayofyear']/365.25)
    df['sin_hour']   = np.sin(2*np.pi*df['hour']/24)
    df['cos_hour']   = np.cos(2*np.pi*df['hour']/24)
    return df

def add_spatial_features(
    df: pd.DataFrame,
    radius_km: float = 10,
    min_samples: int = 3
) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    coords = np.deg2rad(df[['lat','lon']].values)
    tree = BallTree(coords, metric='haversine')
    radius = radius_km / 6371.0
    neighbor_means = []
    for i, point in enumerate(coords):
        idx = tree.query_radius(point.reshape(1,-1), r=radius)[0]
        temps = df.loc[idx, 'temperature'].values
        neighbor_means.append(np.mean(temps) if len(temps)>=min_samples else np.nan)
    df['temp_mean_neighbors'] = neighbor_means
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
    Generuje dokładnie te same cechy, których oczekuje model,
    ale dla jednego punktu (lat, lon, timestamp).
    """
    # 1) ramka z jednym wierszem
    df = pd.DataFrame({
        'timestamp': [timestamp],
        'lat':       [lat],
        'lon':       [lon],
        # dummy (by nie padło validate_schema, jeśli ktoś by odpalił validate)
        'temperature': [np.nan],
        'id':          [0],
    })

    # 2) cechy czasowe
    df = add_time_features(df)

    # 3) cechy przestrzenne
    if use_spatial:
        if not db_path:
            raise FeatureEngineeringError("Do spatial features potrzebna jest ścieżka db_path")
        try:
            df_hist = SQLiteSource(db_path=db_path, table='weather').load()
        except DataSourceError as e:
            raise FeatureEngineeringError(f"Nie udało się wczytać historii: {e}")
        # tylko potrzebujemy kolumny temperature, lat, lon
        df_all = pd.concat([
            df_hist[['temperature','lat','lon']],
            df[['temperature','lat','lon']]
        ], ignore_index=True)
        df_all = add_spatial_features(df_all, radius_km=spatial_radius_km)
        # ostatni wiersz to nasza próbka
        df['temp_mean_neighbors'] = df_all['temp_mean_neighbors'].iloc[-1]

    # ustaw index i usuń niepotrzebne
    return df.set_index('timestamp').drop(columns=['temperature','id'], errors='ignore')

class FeaturePipeline:
    def __init__(
        self,
        use_time: bool = True,
        use_spatial: bool = False,
        spatial_radius_km: float = 10
    ):
        self.use_time = use_time
        self.use_spatial = use_spatial
        self.spatial_radius_km = spatial_radius_km

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Używane tylko do batch-owego featuringu (treningu).
        """
        try:
            df_clean = clean_data(df)
            if self.use_time:
                df_clean = add_time_features(df_clean)
            if self.use_spatial:
                df_clean = add_spatial_features(
                    df_clean,
                    radius_km=self.spatial_radius_km
                )
            return df_clean
        except Exception as e:
            raise FeatureEngineeringError(f"Feature pipeline failed: {e}")
