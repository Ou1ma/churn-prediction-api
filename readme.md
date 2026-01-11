# Customer Churn Prediction API

FastAPI + Random Forest API that predicts bank customer churn, with MLflow tracking.

## Demo
- Docs: `/docs` (Swagger UI)

## What it does
- Trains a Random Forest model (with basic tuning)
- Tracks runs + metrics with MLflow
- Predicts churn + probability + risk level via REST API

## Tech
Python • FastAPI • scikit-learn • MLflow • Pandas/NumPy • Uvicorn

## Dataset
Bank churn dataset (~10k rows).  
Target: **Churn (0/1)**  
Key features: credit score, geography, gender, age, tenure, balance, products, credit card, active member, salary.

## Run locally

```bash
git clone https://github.com/Ou1ma/churn-prediction-api.git
cd churn-prediction-api
pip install -r requirements.txt
