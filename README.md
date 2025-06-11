# Temperature Predictor

This project predicts temperature based on weather data, time, and geographic location.
The LightGBM model is trained on over 1,250,000 historical records, and predictions are served through a simple web application with an interactive map.

## Main Features

- Data ingestion and processing from SQLite database
- Feature engineering (time-based and optionally spatial features)
- LightGBM regression model training (`spatial=True` or `spatial=False` modes)
- Model serialization to disk
- API for serving temperature predictions for given location and date
- Simple web application with interactive map for user input

## Model Summary

- Algorithm: LightGBM
- Training records: ~1,250,000
- Root Mean Squared Error (RMSE): approximately 4.0°C (depending on model variant and data)

## Project Structure

- `weather_forecast/` — data processing, feature engineering, training, and inference modules
- `models/` — directory for saved models (`lgbm_spatial.pkl`, `lgbm_nospatial.pkl`)
- `templates/` — HTML templates for the web application (map frontend)
- `predict.py` — simple CLI interface for predictions
- `app.py` — Flask backend API

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/lipinsks/temperature_predictor.git
cd temperature_predictor
```

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure you have the `data/weather.db` database and trained model saved under `models/`.

5. Run the API server:

```bash
python app.py
```

The service will be available at `http://localhost:5001/` with the interactive map.

## Example CLI usage

```bash
python predict.py 50.0 19.9 2025-06-11T15:00:00
```

## Future Plans

- Improve model accuracy (more data / advanced features - via OpenWeatherStudent pass license)
- Add geographic feature (like rain, wind, etc.)
- Online deployment (Docker / cloud)

---
