from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import FastAPI


# checking curr working directory .a
# import os
# print(os.getcwd())



app = FastAPI(title="Bank churn prediction API")


model = joblib.load("models/bank_churn_model.pkl")
model_cols = joblib.load("models/model_columns.pkl")


class CustomerData(BaseModel):
    CreditScore :int
    Geography:str
    Gender : str
    Age :int
    Tenure:int
    Balance:float
    NumOfProducts:int
    HasCrCard:int
    IsActiveMember:int 
    EstimatedSalary:float




def preprocess_data(data:CustomerData):
    # first convert coming data as dictionary
    input_dict = data.dict()

    # converting input dict to pandas dataframe
    df = pd.DataFrame([input_dict])
    print("before dataframe:" ,df)

    # applying one - hot encoding to the dataFrame just like model
    df = pd.get_dummies(df,columns = ['Geography','Gender'],drop_first = True)
    print

    # make sure ALL expected columns exist, even if this request didn't produce them
    for col in model_cols:
        if col not in df.columns:
            df[col] = 0 
            
    # reorder columns to EXACTLY match what the model was trained on
    df = df[model_cols]
    
    return df



@app.get('/')
def root():
    return {'message':'Churn prediction api is running...'}


@app.get('/health')
def health():
    return {'status':'ok'}

@app.post('/predict')
def predict(data:CustomerData):
    df = preprocess_data(data)

    predict_prob = model.predict_proba(df)[0][1]
    prediction = int(predict_prob > 0.5 )

    return {
        'churn_probablity': round(float(predict_prob),4),
        'churn_prediction' : prediction,
        'risk_level' : "High" if predict_prob > 0.5 else "Low"
    }