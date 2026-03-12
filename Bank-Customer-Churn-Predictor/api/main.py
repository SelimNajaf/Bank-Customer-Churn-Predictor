"""
Bank Churn Prediction API
A FastAPI web service that evaluates customer profiles through a pre-trained 
machine learning pipeline to predict the likelihood of account churn.
"""

import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==========================================
# 1. API INITIALIZATION
# ==========================================
app = FastAPI(
    title='Bank Churn Prediction API',
    description='Predicts whether a bank customer is likely to churn (leave the bank) based on their account metrics and demographics.',
    version='1.0.0'
)


# ==========================================
# 2. MODEL LOADING
# ==========================================
MODEL_PATH = 'trained_model/trained_model.joblib'

try:
    # Load the pre-trained pipeline into memory once during startup
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model from '{MODEL_PATH}'")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    print("Please run 'train_model.py' to generate the model before starting the API.")
    sys.exit(1)


# ==========================================
# 3. PYDANTIC SCHEMAS
# ==========================================
class Customer(BaseModel):
    """Schema representing the expected incoming JSON customer payload."""
    CreditScore: int = Field(..., description="Customer's credit score (e.g., 600)")
    Geography: str = Field(..., description="Country of residence (e.g., 'France', 'Spain', 'Germany')")
    Gender: str = Field(..., description="Customer's gender ('Male' or 'Female')")
    Age: int = Field(..., description="Customer's age in years")
    Tenure: int = Field(..., description="Number of years the customer has been with the bank")
    Balance: float = Field(..., description="Current account balance")
    NumOfProducts: int = Field(..., description="Number of bank products the customer uses (e.g., 1, 2, 3)")
    HasCrCard: int = Field(..., description="Does the customer have a credit card? (1 = Yes, 0 = No)")
    IsActiveMember: int = Field(..., description="Is the customer an active member? (1 = Yes, 0 = No)")
    EstimatedSalary: float = Field(..., description="Estimated annual salary of the customer")

class PredictionResponse(BaseModel):
    """Schema representing the structure of the API response."""
    prediction: int
    probability: float
    status: str


# ==========================================
# 4. PREDICTION ENDPOINT
# ==========================================
@app.post('/predict', response_model=PredictionResponse)
async def predict_churn(data: Customer):
    """
    Receives customer data, processes it through the custom feature engineering 
    and preprocessing pipeline, and predicts the probability of churn.
    """
    try:
        # Step 1: Convert Pydantic JSON payload into a pandas DataFrame
        df = pd.DataFrame([data.model_dump()])

        # Step 2: Execute the pipeline predictions
        # (The pipeline handles FeatureEngineer and Preprocessor automatically)
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        # Step 3: Format and return the response
        return {
            'prediction': int(pred),
            'probability': float(prob),
            'status': 'Churn' if pred == 1 else 'Stay'
        }
        
    except Exception as e:
        # Gracefully handle any unexpected errors during transformation or prediction
        raise HTTPException(status_code=500, detail=f"Prediction processing error: {str(e)}")