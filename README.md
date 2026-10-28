# Flight Fare Prediction API

XGBoost + scikit-learn pipeline served with FastAPI.

## Run
```
pip install -r requirements.txt
python train.py                  # demo: synthetic data (metrics NOT real-world)
python train.py --data itineraries.csv --sample 1000000   # real Expedia CSV, read in chunks
uvicorn app:app --workers 2
python benchmark.py              # measure your own latency / throughput
```
## Design
- One sklearn Pipeline: OneHotEncoder(handle_unknown="ignore") on airports + XGBRegressor -> identical preprocessing at train and inference.
- Metrics: MAE, RMSE, R2 vs mean-fare baseline (saved in model_meta.json).
- API: Pydantic validation, unseen-airport check (422), model loaded once at startup.

## Status
Demo mode uses synthetic data. Real-data results: _fill in after running with the Expedia CSV._

## Roadmap
Time-based split, lead-time/duration features, Docker, endpoint tests.