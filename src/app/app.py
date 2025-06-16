from flask import Flask, request, jsonify, render_template
from weather_forecast.predict import predict
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_temp", methods=["GET", "POST"])
def predict_temp():
    try:
        # 1) Parse inputs
        if request.method == "GET":
            lat = float(request.args.get("lat", ""))
            lon = float(request.args.get("lon", ""))
            ts  = request.args.get("timestamp", "")
        else:  # POST
            data = request.get_json(force=True)
            lat  = float(data.get("lat", ""))
            lon  = float(data.get("lon", ""))
            ts   = data.get("timestamp", "")

        # 2) Validate lat/lon
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude must be between -90 and 90")
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude must be between -180 and 180")

        # 3) Run our predict() and return JSON
        temperature = predict(lat, lon, ts)
        return jsonify({"temperature": float(temperature), "timestamp": ts})

    except ValueError as ve:
        # Input validation errors
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Everything else
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
