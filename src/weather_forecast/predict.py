#!/usr/bin/env python3
# predict.py

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'lgbm_nospatial.pkl'

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df['timestamp']
    df['year']       = ts.dt.year
    df['month']      = ts.dt.month
    df['day']        = ts.dt.day
    df['hour']       = ts.dt.hour
    df['minute']     = ts.dt.minute
    df['dayofyear']  = ts.dt.dayofyear
    df['weekday']    = ts.dt.weekday
    df['sin_day']    = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['cos_day']    = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    df['sin_hour']   = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour']   = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['model']

def predict(lat: float, lon: float, timestamp: str):
    # 1) Przygotuj DataFrame z próbką
    df = pd.DataFrame({
        'timestamp': [pd.to_datetime(timestamp)],
        'lat':       [lat],
        'lon':       [lon]
    })
    # 2) Dodaj cechy czasowe
    df = add_time_features(df)
    df = df.set_index('timestamp')
    # 3) Wczytaj model i wykonaj predykcję
    model = load_model()
    # Upewnij się, że kolumny DataFrame są w tej samej kolejności, czego model się spodziewa:
    feature_names = model.booster_.feature_name()
    X = df[feature_names]
    pred = model.predict(X)
    return f"{pred[0]:.2f}"

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Użycie: python predict.py <lat> <lon> <YYYY-MM-DDThh:mm:ss>")
        sys.exit(1)
    lat  = float(sys.argv[1])
    lon  = float(sys.argv[2])
    ts   = sys.argv[3]
    print(f"Predykcja temperatury: {predict(lat, lon, ts)}")

