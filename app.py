import pandas as pd
from fastapi import FastAPI
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load('model.pkl')

app = FastAPI()

x_test = pd.read_csv("x_test.csv")

x_test = x_test.iloc[:, :10]

@app.get("/predict_all")
def predict_all():
    # Prepare input data from x_test
    predictions = model.predict(x_test)
    return {"predictions": predictions.tolist()}

