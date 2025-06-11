from flask import Flask, request, jsonify, render_template
from weather_forecast.predict import predict

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_temp", methods=["GET", "POST"])
def predict_temp():
    try:
        if request.method == "GET":
            lat = float(request.args.get("lat"))
            lon = float(request.args.get("lon"))
            ts = request.args.get("timestamp")
        else:  # POST
            data = request.json
            lat = float(data.get("lat"))
            lon = float(data.get("lon"))
            ts = data.get("timestamp")

        temperature = float(predict(lat, lon, ts))
        return jsonify({"temperature": temperature, "timestamp": ts})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
