import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Flight Price Prediction API")

# Load once at startup: the pipeline contains the encoder AND the model,
# so training and inference use identical preprocessing.
model = joblib.load("model.pkl")
meta = json.load(open("model_meta.json"))
KNOWN = set(meta["known_airports"])


class FlightRequest(BaseModel):
    startingAirport: str = Field(min_length=3, max_length=3, examples=["JFK"])
    destinationAirport: str = Field(min_length=3, max_length=3, examples=["LAX"])
    isBasicEconomy: bool
    isRefundable: bool
    isNonStop: bool


@app.get("/")
def home():
    return {"message": "API is online! Visit /docs to test predictions.",
            "training_mode": meta["metrics"]["mode"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_price(flight: FlightRequest):
    start, dest = flight.startingAirport.upper(), flight.destinationAirport.upper()
    if start == dest:
        raise HTTPException(422, "startingAirport and destinationAirport must differ")
    unseen = [a for a in (start, dest) if a not in KNOWN]
    if unseen:
        raise HTTPException(422, f"Airport(s) not seen in training: {unseen}. "
                                 f"Known: {sorted(KNOWN)}")
    row = pd.DataFrame([{
        "startingAirport": start, "destinationAirport": dest,
        "isBasicEconomy": int(flight.isBasicEconomy),
        "isRefundable": int(flight.isRefundable),
        "isNonStop": int(flight.isNonStop),
    }])
    try:
        price = float(model.predict(row)[0])
    except Exception as e:
        raise HTTPException(500, f"Model error: {e}")
    return {"predicted_price_USD": round(price, 2)}