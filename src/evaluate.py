import pandas as pd
import joblib
from sklearn.metrics import classification_report


def evaluate():

    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

    model = joblib.load("models/model.pkl")

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate()
