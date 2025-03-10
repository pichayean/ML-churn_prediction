import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Example Data (Replace with your real dataset)
data = pd.DataFrame({
    'age': [55, 60, 45, 70, 50],
    'years_worked': [30, 35, 20, 40, 25],
    'total_income': [60000, 70000, 50000, 80000, 55000],
    'dependents': [2, 3, 1, 4, 2],
    'benefit_amount': [1500, 1800, 1200, 2000, 1400]
})

# Features and Target
X = data.drop(columns=['benefit_amount'])
y = data['benefit_amount']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, 'social_security_model.pkl')
