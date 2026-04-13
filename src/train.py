#!/usr/bin/env python3
"""
ML Pipeline: Train/Val/Test Split, Missing Value Audit & Imputation
Uses SimpleImputer (median / most_frequent) for robustness.
Saves preprocessed data and imputation parameters for later use.
"""
"""
ML Pipeline: Train/Val/Test Split, Missing Value Imputation, Encoding, Scaling,
Two Models (Linear Regression & Random Forest), Comparison, and Saving Best Model.
All transformers fit on train only – no leakage.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# Paths
DATA_PATH = 'data/train.csv'
PROCESSED_DIR = 'processed'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. Load data & select features
# ------------------------------------------------------------
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Full dataset shape: {df.shape}")

features = [
    'LotArea',          # numeric
    'BedroomAbvGr',     # numeric
    'FullBath',         # numeric
    'HalfBath',         # numeric
    'Neighborhood',     # categorical (nominal)
    'OverallQual',      # ordinal (1-10)
    'YearBuilt',        # numeric
    'GarageCars',       # numeric (missing = no garage)
    'KitchenQual',      # ordinal (Ex,Gd,TA,Fa,Po)
    'TotRmsAbvGrd'      # numeric
]
target = 'SalePrice'

X = df[features].copy()
y = df[target].copy()
print(f"Selected {len(features)} features. X shape: {X.shape}")

# ------------------------------------------------------------
# 2. Three-way split (70/15/15)
# ------------------------------------------------------------
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=42
)

print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")

# ------------------------------------------------------------
# 3. Missing value audit (TRAIN only)
# ------------------------------------------------------------
missing_counts = X_train.isnull().sum()
missing_percent = 100 * missing_counts / len(X_train)
missing_table = pd.DataFrame({
    'Missing Count': missing_counts,
    'Missing %': missing_percent
}).sort_values(by='Missing %', ascending=False)

print("\nMissing values in TRAINING set:")
print(missing_table[missing_table['Missing Count'] > 0])

# ------------------------------------------------------------
# 4. Define imputation strategy per column group
# ------------------------------------------------------------
# Columns where missing means "no" or "none" -> fill with 0
numeric_zero_cols = ['GarageCars']

# Columns where missing should be filled with median (numeric)
numeric_median_cols = ['LotArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
                       'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Columns where missing should be filled with mode (categorical)
categorical_mode_cols = ['KitchenQual', 'Neighborhood']

# (Optional: categorical structural 'None' – none needed for our features)

# ------------------------------------------------------------
# 5. Imputation function (fit on train, transform val/test)
# ------------------------------------------------------------
def apply_imputation(X_tr, X_v, X_te,
                     num_zero_cols, num_median_cols, cat_mode_cols):
    """Fit imputers on train, apply to validation and test."""
    X_tr = X_tr.copy()
    X_v = X_v.copy()
    X_te = X_te.copy()

    # Numeric structural -> 0
    for col in num_zero_cols:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].fillna(0)
            X_v[col] = X_v[col].fillna(0)
            X_te[col] = X_te[col].fillna(0)

    # Numeric unknown -> median (fit on train)
    for col in num_median_cols:
        if col in X_tr.columns:
            imp = SimpleImputer(strategy='median')
            X_tr[col] = imp.fit_transform(X_tr[[col]]).ravel()
            X_v[col] = imp.transform(X_v[[col]]).ravel()
            X_te[col] = imp.transform(X_te[[col]]).ravel()

    # Categorical unknown -> mode (fit on train)
    for col in cat_mode_cols:
        if col in X_tr.columns:
            imp = SimpleImputer(strategy='most_frequent')
            X_tr[col] = imp.fit_transform(X_tr[[col]]).ravel()
            X_v[col] = imp.transform(X_v[[col]]).ravel()
            X_te[col] = imp.transform(X_te[[col]]).ravel()

    return X_tr, X_v, X_te

# Apply imputation
X_train_imp, X_val_imp, X_test_imp = apply_imputation(
    X_train, X_val, X_test,
    numeric_zero_cols, numeric_median_cols, categorical_mode_cols
)

print("\nMissing values after imputation:")
print(f"Train: {X_train_imp.isnull().sum().sum()}")
print(f"Val:   {X_val_imp.isnull().sum().sum()}")
print(f"Test:  {X_test_imp.isnull().sum().sum()}")

# ------------------------------------------------------------
# 6. Save preprocessed data & imputation parameters
# ------------------------------------------------------------
X_train_imp.to_csv(f'{PROCESSED_DIR}/X_train.csv', index=False)
X_val_imp.to_csv(f'{PROCESSED_DIR}/X_val.csv', index=False)
X_test_imp.to_csv(f'{PROCESSED_DIR}/X_test.csv', index=False)
y_train.to_csv(f'{PROCESSED_DIR}/y_train.csv', index=False)
y_val.to_csv(f'{PROCESSED_DIR}/y_val.csv', index=False)
y_test.to_csv(f'{PROCESSED_DIR}/y_test.csv', index=False)

# For API reuse: store imputation parameters (only constants and the fitted imputers)
# We'll save the imputers themselves (trained on train) for later use.
imputers = {}
for col in numeric_median_cols:
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train[[col]])
    imputers[f'median_{col}'] = imp
for col in categorical_mode_cols:
    imp = SimpleImputer(strategy='most_frequent')
    imp.fit(X_train[[col]])
    imputers[f'mode_{col}'] = imp
# Also store zero-fill columns list
imputers['numeric_zero_cols'] = numeric_zero_cols
imputers['categorical_mode_cols'] = categorical_mode_cols
imputers['numeric_median_cols'] = numeric_median_cols

with open(f'{PROCESSED_DIR}/imputation_params.pkl', 'wb') as f:
    pickle.dump(imputers, f)

print(f"\nSaved preprocessed data and imputation parameters to '{PROCESSED_DIR}/' folder.")

# Step 3: Encoding, Scaling, and Model Training
# ------------------------------------------------------------
# Reload preprocessed data (already in memory, but we use the imputed DataFrames)
X_train = X_train_imp
X_val = X_val_imp
X_test = X_test_imp

# Define column types for preprocessing
numeric_features = ['LotArea', 'BedroomAbvGr', 'FullBath', 'HalfBath',
                    'OverallQual', 'YearBuilt', 'GarageCars', 'TotRmsAbvGrd']
ordinal_features = ['KitchenQual']
nominal_features = ['Neighborhood']

# Ordinal encoding categories (low to high)
kitchen_categories = [['Po', 'Fa', 'TA', 'Gd', 'Ex']]

# Preprocessing pipelines
numeric_transformer = StandardScaler()
ordinal_transformer = OrdinalEncoder(categories=kitchen_categories)
nominal_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('ord', ordinal_transformer, ordinal_features),
        ('nom', nominal_transformer, nominal_features)
    ])

# Define two models with their pipelines
models = {
    'Linear Regression': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
}
# Train and evaluate on validation set
results = {}
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
    results[name] = {'RMSE': rmse, 'R²': r2}
    print(f"{name:20} | RMSE: {rmse:,.0f} | R²: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline

print("\n" + "="*50)
print(f"Best model: {best_model.named_steps['regressor'].__class__.__name__}")
print(f"Best validation R²: {best_r2:.4f}")
print("="*50)

# Save the best model
with open(f'{MODELS_DIR}/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Best model saved to {MODELS_DIR}/best_model.pkl")

# Final evaluation on test set (only once)
y_pred_test = best_model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
print("\n" + "="*50)
print("FINAL TEST EVALUATION (only once)")
print("="*50)
print(f"Test RMSE: {test_rmse:,.0f}")
print(f"Test R²:  {test_r2:.4f}")
print("="*50)