#!/usr/bin/env python3
"""Model training: load preprocessed data, encode, scale, train two models, save best model."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

PROCESSED_DIR = 'processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Load preprocessed data
X_train = pd.read_csv(f'{PROCESSED_DIR}/X_train.csv')
X_val = pd.read_csv(f'{PROCESSED_DIR}/X_val.csv')
X_test = pd.read_csv(f'{PROCESSED_DIR}/X_test.csv')
y_train = pd.read_csv(f'{PROCESSED_DIR}/y_train.csv').squeeze()
y_val = pd.read_csv(f'{PROCESSED_DIR}/y_val.csv').squeeze()
y_test = pd.read_csv(f'{PROCESSED_DIR}/y_test.csv').squeeze()

# Define column types
numeric_features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
                    'OverallQual', 'YearBuilt', 'GarageCars', 'TotRmsAbvGrd']
ordinal_features = ['KitchenQual']
nominal_features = ['Neighborhood']

kitchen_categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]

# Preprocessor
numeric_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=kitchen_categories)
nominal_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ])

# Models
models = {
    'Linear Regression': Pipeline([('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    'Random Forest': Pipeline([('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
}

# Train and evaluate
best_model = None
best_r2 = -np.inf
print("\n" + "="*50)
print("Model Training and Validation")
print("="*50)

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred_val = pipeline.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2 = r2_score(y_val, y_pred_val)
    print(f"{name:20} | RMSE: {rmse:,.0f} | R²: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

print("\n" + "="*50)
print(f"Best model: {best_model.named_steps['regressor'].__class__.__name__}")
print(f"Best validation R²: {best_r2:.4f}")

# Save best model
with open(f'{MODELS_DIR}/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {MODELS_DIR}/best_model.pkl")

# Final test evaluation
y_pred_test = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
print("\n" + "="*50)
print("FINAL TEST EVALUATION (only once)")
print("="*50)
print(f"Test RMSE: {test_rmse:,.0f}")
print(f"Test R²:  {test_r2:.4f}")