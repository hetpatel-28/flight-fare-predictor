from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Expedia Flight Price Prediction API")

# Load trained model pipeline
model = joblib.load("xgboost_model.pkl")

class FlightRequest(BaseModel):
    startingAirport: str
    destinationAirport: str
    isBasicEconomy: bool
    isRefundable: bool
    isNonStop: bool

@app.get("/")
def home():
    return {"message": "API is online! Visit /docs to test predictions."}

@app.post("/predict")
def predict_price(flight: FlightRequest):
    try:
        data = {
            "startingAirport": flight.startingAirport,
            "destinationAirport": flight.destinationAirport,
            "isBasicEconomy": flight.isBasicEconomy,
            "isRefundable": flight.isRefundable,
            "isNonStop": flight.isNonStop
        }
        
        input_df = pd.DataFrame([data])
        
        # Convert string columns to pandas 'category' data type for XGBoost compatibility
        input_df['startingAirport'] = input_df['startingAirport'].astype('category')
        input_df['destinationAirport'] = input_df['destinationAirport'].astype('category')
        
        prediction = model.predict(input_df)[0]
        return {"predicted_price_USD": round(float(prediction), 2)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Error: {str(e)}")