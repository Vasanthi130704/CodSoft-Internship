from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
feature_names = pickle.load(open("model/features.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "step": float(request.form["step"]),
            "amount": float(request.form["amount"]),
            "oldbalanceOrg": float(request.form["oldbalanceOrg"]),
            "newbalanceOrig": float(request.form["newbalanceOrig"]),
            "oldbalanceDest": float(request.form["oldbalanceDest"]),
            "newbalanceDest": float(request.form["newbalanceDest"]),
            "type_" + request.form["type"]: 1
        }

        df = pd.DataFrame([data])

        # Add missing columns
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        df = df[feature_names]
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)

        if prediction[0] == 1:
            result = "Fraudulent Transaction ❌"
        else:
            result = "Legitimate Transaction ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text=str(e))

if __name__ == "__main__":
    app.run(debug=True)