from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
from typing import Literal

# Load model and preprocessing objects
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/le_geography.pkl', 'rb') as f:
        le_geography = pickle.load(f)
    with open('models/le_gender.pkl', 'rb') as f:
        le_gender = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: Model files not found. Run train.py first!")

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a bank customer will churn based on their profile",
    version="1.0.0"
)

# Define input schema
class CustomerData(BaseModel):
    CreditScore: int = Field(..., ge=300, le=900, description="Customer credit score")
    Geography: Literal["France", "Spain", "Germany"] = Field(..., description="Customer country")
    Gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    Age: int = Field(..., ge=18, le=100, description="Customer age")
    Tenure: int = Field(..., ge=0, le=10, description="Years with the bank")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0 or 1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Is active member (0 or 1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")
    
    class Config:
        json_schema_extra = {
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Male",
                "Age": 35,
                "Tenure": 5,
                "Balance": 50000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 60000.0
            }
        }

# Define output schema
class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    message: str

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(customer: CustomerData):
    """
    Predict customer churn probability
    
    Returns:
    - churn_prediction: 0 (will stay) or 1 (will churn)
    - churn_probability: probability of churning (0-1)
    - risk_level: Low, Medium, or High
    - message: interpretation of the prediction
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    try:
        # Encode categorical variables
        geography_encoded = le_geography.transform([customer.Geography])[0]
        gender_encoded = le_gender.transform([customer.Gender])[0]
        
        # Prepare features in correct order
        features = np.array([[
            customer.CreditScore,
            geography_encoded,
            gender_encoded,
            customer.Age,
            customer.Tenure,
            customer.Balance,
            customer.NumOfProducts,
            customer.HasCrCard,
            customer.IsActiveMember,
            customer.EstimatedSalary
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
            message = "Customer is likely to stay. Low churn risk."
        elif probability < 0.6:
            risk_level = "Medium"
            message = "Customer shows some churn indicators. Consider retention strategies."
        else:
            risk_level = "High"
            message = "Customer is at high risk of churning. Immediate action recommended."
        
        return PredictionResponse(
            churn_prediction=int(prediction),
            churn_probability=round(float(probability), 4),
            risk_level=risk_level,
            message=message
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/feature-importance")
def get_feature_importance():
    """Get feature importance from the model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_names = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
    ]
    
    importances = model.feature_importances_
    
    feature_importance = {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, importances)
    }
    
    # Sort by importance
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    return {"feature_importance": feature_importance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)