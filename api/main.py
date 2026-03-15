from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import skops.io as sio
import numpy as np
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
from config import MODELS_DIR, PREPROCESSOR_PATH


app = FastAPI()

model = sio.load(MODELS_DIR / "Logistic_Regression.skops")
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH / "preprocessor.joblib")
except FileNotFoundError:
    preprocessor = None

@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Churn Prediction</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            textarea { width: 400px; height: 300px; font-family: monospace; }
        </style>
    </head>
    <body>
        <h2>Customer Churn Prediction</h2>
        <p>Enter features as JSON object:</p>
        <textarea id="features">
{
  "tenure": 5,
  "MonthlyCharges": 50.0,
  "TotalCharges": 250.0,
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check"
}
        </textarea>
        <br><br>
        <button onclick="predict()">Predict</button>
        <h3 id="result"></h3>

        <script>
        async function predict() {
            let input = document.getElementById("features").value;
            let features;
            try {
                features = JSON.parse(input);
            } catch (e) {
                alert("Invalid JSON");
                return;
            }

            try {
                const response = await fetch("/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ "features": features })
                });

                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`Server error: ${response.status} ${response.statusText}\\n${text}`);
                }

                const data = await response.json();
                document.getElementById("result").innerHTML =
                    "Prediction: " + data.prediction +
                    "<br>Probability: " + data.churn_probability;

            } catch (err) {
                console.error(err);
                document.getElementById("result").innerHTML = "Error: " + err.message;
            }
        }
        </script>
    </body>
    </html>
    """

@app.post("/")
async def predict(request: Request):
    body = await request.json()
    features_dict = body["features"]

    df = pd.DataFrame([features_dict])
    
    # 1. Basic Cleaning
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    # 2. Binary Encoding
    yes_no_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]
    for col in yes_no_cols:
        if col in df.columns:
            if col == "gender":
                df[col] = df[col].map({"Male": 1, "Female": 0}).fillna(0).astype(int)
            else:
                # If it's already an int (e.g. SeniorCitizen), skip map
                if df[col].dtype == object:
                    df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # 3. Apply transformer
    data = preprocessor.transform(df)

    prob = model.predict_proba(data)[0][1]

    return {
        "churn_probability": float(prob),
        "prediction": "Yes" if prob > 0.5 else "No",
    }

