# Expedia Flight Price Prediction API ✈️

An end-to-end machine learning service that predicts flight prices using XGBoost and FastAPI, trained on 1,000,000 production flight records from Expedia.

## 📌 Project Highlights
- **Data Engineering:** Processed and sampled 1M rows from a 30 GB dataset using Pandas to optimize memory usage.
- **Machine Learning:** Built a Scikit-Learn pipeline featuring One-Hot Encoding for route categorical variables (`startingAirport`, `destinationAirport`) and an XGBoost Regressor.
- **REST API Deployment:** Encapsulated model inference in FastAPI with automated Pydantic request validation and high-throughput Uvicorn server execution.

## 🚀 How to Run Locally

1. Clone the repository:
   ```bash
   git clone <YOUR-GITHUB-REPO-URL>
   cd flight_api