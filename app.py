import pandas as pd
from fastapi import FastAPI, Request ,  File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import shap
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


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




    data.drop(['customerID'], axis = 1, inplace = True)
    data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] = data[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']].replace({'No internet service': 'No'})
    data['MultipleLines'] = data['MultipleLines'].replace({'No phone service': 'No'})
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna(subset=['TotalCharges'])
    data['tenure_category'] = pd.qcut(data['tenure'], q=4, labels=['New', 'Short-term', 'Mid-term', 'Long-term'])

    tenure_encoder = LabelEncoder()
    data['tenure_category'] = tenure_encoder.fit_transform(data['tenure_category'])

    high_charge_threshold = data['MonthlyCharges'].quantile(0.75)
    data['high_monthly_charge'] = (data['MonthlyCharges'] > high_charge_threshold).astype(int)

    data.drop(['tenure'], axis = 1, inplace = True)
    data['SeniorCitizen'] = data['SeniorCitizen'].astype('object')


    binary_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService','MultipleLines',  
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',  'PaperlessBilling', 'Churn'
    ]
    one_hot_encode_cols = ['InternetService','Contract', 'PaymentMethod', 'tenure_category']

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), one_hot_encode_cols),
            ('ordinal', OrdinalEncoder(), binary_cols)
        ],
        remainder='passthrough'
    )

    data_encoded = preprocessor.fit_transform(data)

    one_hot_columns = preprocessor.named_transformers_['onehot'].get_feature_names_out(one_hot_encode_cols)
    binary_columns = binary_cols
    numeric_columns = ['MonthlyCharges', 'TotalCharges', 'high_monthly_charge']

    final_columns_adjusted = list(one_hot_columns) + binary_columns + numeric_columns

    # Create the final DataFrame with updated column names
    data_encoded = pd.DataFrame(data_encoded, columns=final_columns_adjusted)

    scaler = StandardScaler()
    data_encoded[['MonthlyCharges', 'TotalCharges', 'high_monthly_charge']] = scaler.fit_transform(data_encoded[['MonthlyCharges', 'TotalCharges', 'high_monthly_charge']])

    X = data_encoded.drop(['Churn'], axis = 1)
    X= X.values
        
        
    # features_for_prediction = data.iloc[0, :10] 
    # features_for_prediction = features_for_prediction.values.reshape(1, -1)
   
    prediction = model.predict(X)
    print(prediction)
    
    return {"prediction": prediction}