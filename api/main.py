from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")


@app.post("/predict")
def predict(features: list):

    data = np.array(features).reshape(1, -1)
    prob = model.predict_proba(data)[0][1]

    return {
        "churn_probability": float(prob),
        "prediction": "Yes" if prob > 0.5 else "No",
    }
