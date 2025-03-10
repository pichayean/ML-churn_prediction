from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = FastAPI()

MODEL_FILE = "model.pkl"

# Check if a trained model exists
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = None

# Define the input format for prediction
class CustomerInput(BaseModel):
    monthly_spend: float
    tenure: int
    support_tickets: int
    last_rating: int

# Define the input format for training
class TrainData(BaseModel):
    data: list  # List of dictionaries with training data

@app.post("/train/")
def train_model(train_data: TrainData):
    global model

    # Convert list of dicts to Pandas DataFrame
    df = pd.DataFrame(train_data.data)

    if 'churn' not in df.columns:
        return {"error": "Dataset must include 'churn' column"}

    # Split dataset
    X = df.drop(columns=['churn'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model
    joblib.dump(model, MODEL_FILE)

    return {"message": "Model trained successfully", "accuracy": accuracy}

@app.post("/predict/")
def predict_churn(customer: CustomerInput):
    if model is None:
        return {"error": "Model not trained yet. Please train first."}

    # Convert input to NumPy array
    data = np.array([[customer.monthly_spend, customer.tenure, customer.support_tickets, customer.last_rating]])

    # Make prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]  # Probability of churn

    return {"churn": bool(prediction), "confidence": round(probability * 100, 2)}

