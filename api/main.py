from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import skops.io as sio
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

app = FastAPI()

model = sio.load(MODELS_DIR / "logistic_regression.skops")


@app.get("/", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <head>
        <title>Churn Prediction</title>
    </head>

    <body>
        <h2>Customer Churn Prediction</h2>

        <p>Enter features separated by comma (example: 600,40,3,50000,...)</p>

        <input type="text" id="features" style="width:400px">

        <br><br>

        <button onclick="predict()">Predict</button>

        <h3 id="result"></h3>

        <script>
        async function predict() {
            let input = document.getElementById("features").value;
            let features = input.split(",").map(Number);

            try {
                const response = await fetch("/", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ "features": features })
                });

                if (!response.ok) {
                    // Server returned error (500, 400, etc.)
                    const text = await response.text();
                    throw new Error(`Server error: ${response.status} ${response.statusText}\n${text}`);
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

    body = await request.json()  # read JSON body
    features = body["features"]

    data = np.array(features).reshape(1, -1)

    prob = model.predict_proba(data)[0][1]

    return {
        "churn_probability": float(prob),
        "prediction": "Yes" if prob > 0.5 else "No",
    }
