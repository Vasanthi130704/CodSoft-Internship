from flask import Flask, render_template, request
import pickle
import pandas as pd
import os
import numpy as np

# Create Flask app
app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "feature.pkl")

# Load model files
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

with open(FEATURE_PATH, "rb") as f:
    features = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:

        # Get form inputs
        amt = float(request.form.get("amt"))
        gender = request.form.get("gender")
        city_pop = float(request.form.get("city_pop"))
        lat = float(request.form.get("lat"))
        long = float(request.form.get("long"))
        merch_lat = float(request.form.get("merch_lat"))
        merch_long = float(request.form.get("merch_long"))
        unix_time = float(request.form.get("unix_time"))

        # Encode gender
        gender_val = 1 if gender == "F" else 0

        # Feature Engineering
        # Distance between customer & merchant
        distance = np.sqrt(
            (lat - merch_lat) ** 2 +
            (long - merch_long) ** 2
        )

        # Transaction hour
        hour = pd.to_datetime(unix_time, unit="s").hour
        # Create dataframe
        data = {
            "amt": amt,
            "gender": gender_val,
            "city_pop": city_pop,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long,
            "unix_time": unix_time,
            "distance": distance,
            "hour": hour
        }

        df = pd.DataFrame([data])

        # Ensure feature order
        df = df[features]

        # Scale input
        scaled = scaler.transform(df)

        # Fraud probability
        fraud_prob = model.predict_proba(scaled)[0][1]

        # Decision threshold
        if fraud_prob > 0.21:
            result = f"Fraudulent Transaction ❌ (Probability: {fraud_prob:.2f})"
        else:
            result = f"Legitimate Transaction ✅ (Probability: {fraud_prob:.2f})"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)