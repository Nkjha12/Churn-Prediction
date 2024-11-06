import pandas as pd
from fastapi import FastAPI, Request ,  File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import io

app = FastAPI()
model = joblib.load('model.pkl')


x_test = pd.read_csv("x_test.csv")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    contents = await file.read()
    data = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    
    
    features_for_prediction = data.iloc[0, :10] 
    features_for_prediction = features_for_prediction.values.reshape(1, -1)
   
    prediction = model.predict(features_for_prediction)
    
    return {"prediction": int(prediction[0])}